"""
Siamese KPConv for per-point change detection on a bi-temporal cube pair
(Section 2 model `siamese_kpconv`, a VARIABLE-N model). This is the de Gelis et al.
Siamese-KPConv mechanism, the SOTA reference on Urb3DCD: a shared KPConv encoder
embeds both cubes, then the encoded t0/t1 features are differenced point-to-point and
fed to a seg head. We keep the project's proven NN feature-diff (nearest_t0_features
idea): for every t1 point gather its nearest t0 point's encoded feature in the shared
normalised frame and take feat_t1 - matched_t0. That local correspondence is what
localises change (the Task 5 lesson shared by every model in this project).

Variable-N: this model consumes the ragged FPS-1024 cube clouds (lengths 256..1024,
no padding, every point real) for cube-pipeline parity with the other models. train.py
and predict.py run it ONE CUBE AT A TIME (no batch/mask), routing each cube's t1 logits
through the existing per-cube NN-propagation. Forward contract:
    forward(xyz0, xyz1) -> per-point t1 logits [N1, 2]
with xyz0 [N0, 3], xyz1 [N1, 3] single-cube normalised coords.

Self-contained pure-torch KPConv (no torch-points3d / CUDA ops, which are fragile on
Kaggle). The operator is the standard rigid KPConv: a fixed set of kernel points in a
local ball, each neighbour's influence on a kernel point is the linear correlation
max(0, 1 - dist/sigma), and the output is the sum over kernel points of (influence-
weighted neighbour features) passed through a per-kernel-point weight matrix. The
encoder stacks a few KPConv blocks at native resolution so every point keeps a feature
for the t0->t1 match. All neighbour searches run on the normalised [-1,1] cube coords
(small magnitude, float32-safe); the original metric coords are only touched later, in
predict.py's NN-propagation, which already centres them.
"""

import numpy as np
import torch
import torch.nn as nn


def make_kernel_points(radius):
    """
    Fixed rigid kernel-point layout: the origin, the six axis directions, and the eight
    cube-corner directions, scaled to `radius`. 15 well-spread points, symmetric, good
    enough for a self-contained KPConv (no repulsive optimisation needed). Returns a
    [15, 3] float32 array in the normalised cube frame.
    """
    points = [[0.0, 0.0, 0.0]]                            # centre kernel point
    for axis in range(3):                                 # six axis directions
        for sign in (+1.0, -1.0):
            p = [0.0, 0.0, 0.0]
            p[axis] = sign
            points.append(p)
    for sx in (+1.0, -1.0):                               # eight cube corners (normalised)
        for sy in (+1.0, -1.0):
            for sz in (+1.0, -1.0):
                norm = np.sqrt(3.0)
                points.append([sx / norm, sy / norm, sz / norm])
    return (np.array(points, dtype=np.float32) * radius)


class KPConv(nn.Module):
    """
    One rigid kernel-point convolution. Maps support features at support points to new
    features at query points using kNN neighbourhoods and fixed kernel points. For the
    encoder we use it as a self-convolution (query == support).
    """

    def __init__(self, in_channel, out_channel, k=16, radius=0.15, sigma=0.15):
        super().__init__()
        self.k = k                                        # neighbours per query point
        self.sigma = sigma                                # kernel influence radius (linear correlation)
        kernel = make_kernel_points(radius)               # [K, 3] fixed kernel points
        self.num_kernel = kernel.shape[0]
        # Kernel points are fixed geometry, not learned -> store as a buffer.
        self.register_buffer("kernel_points", torch.from_numpy(kernel))
        # One weight matrix per kernel point: [K, in, out]. This is what KPConv learns.
        self.weight = nn.Parameter(torch.empty(self.num_kernel, in_channel, out_channel))
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))   # standard linear-layer init

    def forward(self, query_xyz, support_xyz, support_feat):
        # query_xyz [Nq, 3]; support_xyz [Ns, 3]; support_feat [Ns, Cin].
        num_neighbours = min(self.k, support_xyz.shape[0])    # tiny cubes may have < k points
        dist = torch.cdist(query_xyz, support_xyz)            # [Nq, Ns] normalised frame (safe)
        neighbour_idx = dist.topk(num_neighbours, dim=1, largest=False).indices   # [Nq, k]
        # Neighbour positions relative to their query point, and their features.
        neighbour_xyz = support_xyz[neighbour_idx] - query_xyz.unsqueeze(1)   # [Nq, k, 3]
        neighbour_feat = support_feat[neighbour_idx]                          # [Nq, k, Cin]
        # Distance from each neighbour to each kernel point -> linear-correlation influence.
        diff = neighbour_xyz.unsqueeze(2) - self.kernel_points.view(1, 1, self.num_kernel, 3)
        kernel_dist = torch.linalg.norm(diff, dim=3)          # [Nq, k, K]
        influence = (1.0 - kernel_dist / self.sigma).clamp(min=0.0)   # [Nq, k, K]
        # Aggregate neighbour features per kernel point (influence-weighted sum over neighbours).
        weighted = torch.einsum("nkK,nki->nKi", influence, neighbour_feat)   # [Nq, K, Cin]
        # Sum over kernel points of (aggregated feature @ that kernel point's weight matrix).
        out = torch.einsum("nKi,Kio->no", weighted, self.weight)             # [Nq, Cout]
        return out


class KPConvBlock(nn.Module):
    """KPConv self-convolution + BatchNorm + ReLU on a single cube's points."""

    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.kpconv = KPConv(in_channel, out_channel)
        self.bn = nn.BatchNorm1d(out_channel)             # BN over the cube's points
        self.relu = nn.ReLU()

    def forward(self, xyz, feat):
        # Self-convolution: query and support are the same cube points.
        out = self.kpconv(xyz, xyz, feat)                 # [N, Cout]
        out = self.bn(out)                                # normalise over the cube's points
        return self.relu(out)


class SharedKPConvEncoder(nn.Module):
    """
    Shared KPConv encoder: four KPConv blocks at native resolution, so every point keeps
    a 128-d feature for the t0->t1 match. Input feature is a constant 1 (KPConv reads
    geometry through its kernel points, the standard first-layer input). Returns a
    per-point feature [N, 128].
    """

    def __init__(self):
        super().__init__()
        self.block1 = KPConvBlock(1, 64)
        self.block2 = KPConvBlock(64, 64)
        self.block3 = KPConvBlock(64, 128)
        self.block4 = KPConvBlock(128, 128)

    def forward(self, xyz):
        # Constant input feature (geometry-only first layer, as in KPConv).
        feat = torch.ones(xyz.shape[0], 1, device=xyz.device)   # [N, 1]
        feat = self.block1(xyz, feat)                            # [N, 64]
        feat = self.block2(xyz, feat)                            # [N, 64]
        feat = self.block3(xyz, feat)                            # [N, 128]
        feat = self.block4(xyz, feat)                            # [N, 128]
        return feat


def nearest_t0_feature_single(xyz0, xyz1, feat0):
    """
    Per-cube (no batch, no mask) version of the t1->t0 nearest-feature gather: for each
    t1 point return its nearest t0 point's feature in the shared normalised frame. NN is
    used only to pick the correspondence index; the distance value is discarded (the
    change signal is the learned feature difference, so no Chamfer distance leaks in).
    xyz0 [N0, 3], xyz1 [N1, 3], feat0 [N0, C] -> matched [N1, C].
    """
    dist = torch.cdist(xyz1, xyz0)                        # [N1, N0] normalised frame (safe)
    nn_index = dist.argmin(dim=1)                         # [N1] nearest t0 point per t1 point
    return feat0[nn_index]                                # [N1, C] matched t0 features


class SiameseKPConv(nn.Module):
    """Siamese KPConv change head. forward(xyz0, xyz1) -> per-point t1 logits [N1, 2]."""

    def __init__(self, num_classes=2):
        super().__init__()
        self.encoder = SharedKPConvEncoder()
        # Seg head input per t1 point = t1 local (128) + change diff (128)
        # + t1 global (128) + t0 global (128) = 512 channels. Linear layers (per-point).
        self.seg_mlp = nn.Sequential(
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, num_classes),                  # final layer -> raw logits, no activation
        )

    def forward(self, xyz0, xyz1):
        # Shared encoder on both cubes (weights tied -> Siamese). One cube at a time.
        feat0 = self.encoder(xyz0)                         # [N0, 128] t0 per-point features
        feat1 = self.encoder(xyz1)                         # [N1, 128] t1 per-point features
        matched_t0 = nearest_t0_feature_single(xyz0, xyz1, feat0)   # [N1, 128]
        change_diff = feat1 - matched_t0                   # [N1, 128] learned per-point change signal
        # Global cube descriptors (max over points), broadcast to every t1 point.
        global1 = feat1.max(dim=0).values                  # [128] t1 global
        global0 = feat0.max(dim=0).values                  # [128] t0 global
        num_points = xyz1.shape[0]
        g1 = global1.unsqueeze(0).expand(num_points, -1)   # [N1, 128]
        g0 = global0.unsqueeze(0).expand(num_points, -1)   # [N1, 128]
        seg_in = torch.cat([feat1, change_diff, g1, g0], dim=1)   # [N1, 512]
        return self.seg_mlp(seg_in)                        # [N1, 2] per-point logits
