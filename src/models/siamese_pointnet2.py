"""
Siamese PointNet++ for per-point change detection on a bi-temporal cube pair
(Section 2 model `siamese_pointnet2`, a fixed-N model).

Same hard-won design as siamese_pointnet (the Task 5 template): a SHARED encoder
embeds both cubes, then for every t1 point we gather its nearest t0 point's LEARNED
feature in the shared normalised frame and feed the difference (feat_t1 - matched_t0)
to the seg head. That local correspondence is what localises change; a global-only
head fails (AUC ~0.6, below ICP). The ONLY thing that changes here vs siamese_pointnet
is the encoder: a PointNet++ set-abstraction (down) + feature-propagation (up) stack
instead of a flat per-point MLP, so the per-point features see a multi-scale local
neighbourhood. The NN feature-diff head is reused verbatim (nearest_t0_features).

This is a self-contained PointNet++ SSG (single-scale grouping) in pure torch with
kNN grouping (no ball-query radius to tune, no CUDA ops). Masking for the fixed-N
padding lives only at the FIRST set-abstraction layer: padded slots are never picked
as FPS centroids and never grouped as neighbours. Every later level operates on real
sampled centroids only, and feature-propagation back to full resolution interpolates
from those real centroids, so padded query points get throwaway features that the loss
mask discards. The internal NN/cdist all run on normalised [-1,1] coords, so the
float32 cancellation that bit Task 5 (cdist on million-scale coords) cannot occur here.

Uniform fixed-N forward contract (identical to siamese_pointnet so train.py /
predict.py call every fixed-N model the same way):
    forward(xyz0, xyz1, mask0, mask1) -> per-point t1 logits [B, N, 2]
"""

import torch
import torch.nn as nn

# Reuse the proven t1->t0 nearest-neighbour feature-difference helper (the Task 5
# mechanism). Importing keeps the change signal identical across all fixed-N models.
from .siamese_pointnet import nearest_t0_features


def index_points(points, idx):
    """
    Gather points by index along the point dimension.
    points [B, N, C]; idx [B, S] or [B, S, K] (long) -> [B, S, C] or [B, S, K, C].
    """
    batch_size = points.shape[0]
    index_shape = idx.shape[1:]                           # (S,) or (S, K)
    # Per-batch index broadcast to the same trailing shape as idx for advanced indexing.
    batch_index = torch.arange(batch_size, device=points.device)
    batch_index = batch_index.view(batch_size, *([1] * len(index_shape)))
    batch_index = batch_index.expand(batch_size, *index_shape)
    return points[batch_index, idx]


def masked_farthest_point_sample(xyz, mask, npoint):
    """
    Farthest point sampling that never selects a padded slot. Returns centroid
    indices [B, npoint] into xyz. xyz [B, N, 3], mask [B, N] bool (True = real).

    distance holds, per point, the squared distance to the nearest centroid chosen
    so far; padded points are pinned to -1 so the argmax (farthest) never picks them
    while any real point still has a non-negative distance. Seed is the first real
    point of each cube. npoint is kept <= the 256-point minimum cube size, so a real
    cube always has enough real points to fill the centroid set.
    """
    batch_size, num_points, _ = xyz.shape
    device = xyz.device
    centroids = torch.zeros(batch_size, npoint, dtype=torch.long, device=device)
    distance = torch.full((batch_size, num_points), 1e10, device=device)
    distance = distance.masked_fill(~mask, -1.0)          # padded never wins the farthest argmax
    farthest = mask.float().argmax(dim=1)                 # first real point per cube as the seed
    batch_index = torch.arange(batch_size, device=device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid_xyz = xyz[batch_index, farthest].unsqueeze(1)   # [B, 1, 3] current centroid
        dist = ((xyz - centroid_xyz) ** 2).sum(dim=2)            # [B, N] squared distance
        update = (dist < distance) & mask                        # shrink only real points' distance
        distance[update] = dist[update]
        farthest = distance.argmax(dim=1)                        # next farthest real point
    return centroids


def masked_knn(centroid_xyz, xyz, mask, k):
    """
    k nearest REAL neighbours of each centroid. centroid_xyz [B, S, 3], xyz [B, N, 3],
    mask [B, N] -> neighbour indices [B, S, k] into xyz. Padded points are pushed to a
    huge distance so they are never grouped. cdist runs on normalised coords (safe).
    """
    dist = torch.cdist(centroid_xyz, xyz)                 # [B, S, N] in the normalised frame
    dist = dist.masked_fill(~mask.unsqueeze(1), 1e10)     # padded points cannot be neighbours
    return dist.topk(k, dim=2, largest=False).indices     # [B, S, k] nearest real points


class SetAbstraction(nn.Module):
    """
    One PointNet++ set-abstraction level (single-scale, kNN grouping). Samples npoint
    centroids, groups k neighbours, runs a shared MLP on [relative xyz (+ feature)] and
    max-pools over the neighbourhood. mask selects valid input points (only meaningful at
    the first level; later levels pass an all-True mask over real centroids).
    """

    def __init__(self, npoint, k, in_channel, mlp_channels):
        super().__init__()
        self.npoint = npoint
        self.k = k
        layers = []
        last = in_channel
        for out_channel in mlp_channels:
            # Conv2d with kernel 1 is the shared per-(point,neighbour) MLP.
            layers += [nn.Conv2d(last, out_channel, 1), nn.BatchNorm2d(out_channel), nn.ReLU()]
            last = out_channel
        self.mlp = nn.Sequential(*layers)

    def forward(self, xyz, feat, mask):
        # xyz [B, N, 3]; feat [B, N, C] or None; mask [B, N] bool.
        centroid_idx = masked_farthest_point_sample(xyz, mask, self.npoint)   # [B, npoint]
        new_xyz = index_points(xyz, centroid_idx)                             # [B, npoint, 3]
        neighbour_idx = masked_knn(new_xyz, xyz, mask, self.k)                # [B, npoint, k]
        # Relative neighbour coordinates centre the local patch on its centroid.
        grouped_xyz = index_points(xyz, neighbour_idx) - new_xyz.unsqueeze(2)  # [B, npoint, k, 3]
        if feat is None:
            grouped = grouped_xyz                                             # [B, npoint, k, 3]
        else:
            grouped_feat = index_points(feat, neighbour_idx)                  # [B, npoint, k, C]
            grouped = torch.cat([grouped_xyz, grouped_feat], dim=3)           # [B, npoint, k, 3+C]
        grouped = grouped.permute(0, 3, 1, 2)                                 # [B, Cin, npoint, k]
        grouped = self.mlp(grouped)                                          # [B, Cout, npoint, k]
        new_feat = grouped.max(dim=3).values                                # [B, Cout, npoint] pool over k
        return new_xyz, new_feat.transpose(1, 2)                            # xyz, [B, npoint, Cout]


class FeaturePropagation(nn.Module):
    """
    One PointNet++ feature-propagation level: interpolate the coarse-level features up
    to the finer query points by inverse-distance weighting of their 3 nearest support
    points, concatenate the skip features, and run a shared 1x1 MLP. Support points are
    always real sampled centroids, so no mask is needed (padded query points just get
    throwaway features the loss later discards).
    """

    def __init__(self, in_channel, mlp_channels):
        super().__init__()
        layers = []
        last = in_channel
        for out_channel in mlp_channels:
            layers += [nn.Conv1d(last, out_channel, 1), nn.BatchNorm1d(out_channel), nn.ReLU()]
            last = out_channel
        self.mlp = nn.Sequential(*layers)

    def forward(self, query_xyz, support_xyz, skip_feat, support_feat):
        # query_xyz [B, Nq, 3]; support_xyz [B, Ns, 3]; support_feat [B, Ns, C].
        dist = torch.cdist(query_xyz, support_xyz)        # [B, Nq, Ns] normalised frame (safe)
        knn = dist.topk(3, dim=2, largest=False)          # 3 nearest support points per query
        d3 = knn.values.clamp(min=1e-8)                   # [B, Nq, 3] guard divide-by-zero
        weight = 1.0 / d3
        weight = weight / weight.sum(dim=2, keepdim=True)  # [B, Nq, 3] inverse-distance weights
        gathered = index_points(support_feat, knn.indices)  # [B, Nq, 3, C]
        interpolated = (gathered * weight.unsqueeze(3)).sum(dim=2)  # [B, Nq, C]
        if skip_feat is not None:
            interpolated = torch.cat([interpolated, skip_feat], dim=2)   # [B, Nq, C+Cskip]
        out = self.mlp(interpolated.transpose(1, 2))      # [B, Cout, Nq]
        return out.transpose(1, 2)                         # [B, Nq, Cout]


class SharedPointNet2Encoder(nn.Module):
    """
    Shared PointNet++ encoder: two set-abstraction levels down (1024 -> 256 -> 64) and
    two feature-propagation levels back up to full resolution. Returns a per-point
    feature [B, 128, N] (channels-first, for the NN feature-diff head) and a global cube
    descriptor [B, 256] (max over the coarse 64-centroid features).
    """

    def __init__(self):
        super().__init__()
        # npoint=256 stays within the 256-point per-cube minimum so masked FPS always
        # finds enough real centroids even in the smallest take-all cube.
        self.sa1 = SetAbstraction(npoint=256, k=32, in_channel=3, mlp_channels=[64, 64, 128])
        self.sa2 = SetAbstraction(npoint=64, k=32, in_channel=3 + 128, mlp_channels=[128, 128, 256])
        self.fp2 = FeaturePropagation(in_channel=256 + 128, mlp_channels=[256, 128])
        self.fp1 = FeaturePropagation(in_channel=128, mlp_channels=[128, 128])

    def forward(self, xyz, mask):
        # xyz [B, N, 3] normalised input; mask [B, N] bool (True = real).
        l1_xyz, l1_feat = self.sa1(xyz, None, mask)       # [B, 256, 3], [B, 256, 128]
        # All sampled centroids are real, so later levels use an all-True mask.
        all_real = torch.ones(l1_xyz.shape[0], l1_xyz.shape[1], dtype=torch.bool, device=xyz.device)
        l2_xyz, l2_feat = self.sa2(l1_xyz, l1_feat, all_real)   # [B, 64, 3], [B, 64, 256]
        # Propagate coarse features back up: l2 -> l1 (skip l1_feat) -> full resolution.
        l1_feat = self.fp2(l1_xyz, l2_xyz, l1_feat, l2_feat)    # [B, 256, 128]
        l0_feat = self.fp1(xyz, l1_xyz, None, l1_feat)          # [B, N, 128] per-point feature
        point_feat = l0_feat.transpose(1, 2)                    # [B, 128, N] channels-first
        global_feat = l2_feat.max(dim=1).values                # [B, 256] global cube descriptor
        return point_feat, global_feat


class SiamesePointNet2(nn.Module):
    """Siamese PointNet++ change head. forward -> per-point t1 logits [B, N, 2]."""

    def __init__(self, num_classes=2):
        super().__init__()
        self.encoder = SharedPointNet2Encoder()
        # Seg head input per t1 point = t1 local (128) + change diff (128)
        # + t1 global (256) + t0 global (256) = 768 channels.
        self.seg_mlp = nn.Sequential(
            nn.Conv1d(768, 256, 1), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Conv1d(256, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, num_classes, 1),               # final layer -> raw logits, no activation
        )

    def forward(self, xyz0, xyz1, mask0, mask1):
        # Shared encoder on both cubes (weights tied -> Siamese).
        feat0, global0 = self.encoder(xyz0, mask0)        # t0 per-point features + global
        feat1, global1 = self.encoder(xyz1, mask1)        # t1 per-point features + global
        # Match each t1 point to its nearest t0 point and take the learned difference.
        matched_t0 = nearest_t0_features(xyz0, xyz1, feat0, mask0)   # [B, 128, N]
        change_diff = feat1 - matched_t0                  # [B, 128, N] learned per-point change signal
        num_points = xyz1.shape[1]
        # Broadcast both global descriptors to every t1 point for cube-level context.
        g1 = global1.unsqueeze(2).expand(-1, -1, num_points)   # [B, 256, N] t1 global
        g0 = global0.unsqueeze(2).expand(-1, -1, num_points)   # [B, 256, N] t0 global
        seg_in = torch.cat([feat1, change_diff, g1, g0], dim=1)  # [B, 768, N]
        logits = self.seg_mlp(seg_in)                            # [B, 2, N]
        return logits.transpose(1, 2)                            # [B, N, 2] back to points-first
