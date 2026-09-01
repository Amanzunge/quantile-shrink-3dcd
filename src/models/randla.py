"""
RandLA-Net style Siamese encoder for per-point change detection on a bi-temporal
cube pair (Section 2 model `randla`, a VARIABLE-N model). RandLA-Net is the
large-scale-segmentation reference whose signature is efficiency via three
ingredients: (a) Local Spatial Encoding, (b) Attentive Pooling, and (c) Random
Sampling for downsampling in an encoder-decoder U-Net. We keep all three in a
compact two-level U-Net and reuse the project's proven NN feature-diff head (the
Task 5 lesson shared by every model): for every t1 point gather its nearest t0
point's encoded feature in the shared normalised frame and take feat_t1 - matched_t0.
That local correspondence is what localises change.

RandLA is INCLUDED on MS/HKCD and excluded from LD (insufficient density there). It
is the large-N model: it consumes the FPS-4096 cube directly. Like siamese_kpconv it
is VARIABLE-N (ragged native clouds, every point real, no padding/mask); train.py and
predict.py run it ONE CUBE AT A TIME and route each cube's t1 logits through the
existing per-cube NN-propagation. Forward contract (identical to siamese_kpconv):
    forward(xyz0, xyz1) -> per-point t1 logits [N1, 2]
with xyz0 [N0, 3], xyz1 [N1, 3] single-cube normalised coords.

Self-contained pure-torch (no torch-points3d / CUDA ops, which are fragile on Kaggle).
Every neighbour search runs on the normalised [-1, 1] cube coords (small magnitude,
float32-safe); the original metric coords are only touched later, in predict.py's
NN-propagation, which already centres them.

Head/loss parity with the other Siamese models (Section 7, architecture is the only
variable): the encoder produces a 128-d per-point feature and the SAME seg head as
siamese_kpconv runs over cat[feat1, feat1 - matched_t0, global1, global0] = 512 ->
256 -> 128 -> 2. The RandLA encoder is therefore the only thing that differs.

Note on Random Sampling at predict time: RandLA samples points randomly inside the
network at train AND test (every input point still gets a feature through the decoder
upsampling, so the per-point output is well defined). predict.py seeds torch before
the pass so the saved val/test npz is reproducible.
"""

import torch
import torch.nn as nn


def knn_index(xyz, k):
    """k nearest neighbours of every point (itself included) by Euclidean distance in the
    normalised cube frame (values in [-1, 1], so cdist is float32-safe). Returns idx [N, k].
    k is clamped to the cloud size so tiny cubes (down to 256 points, or 16 at the coarsest
    encoder level) still work."""
    k = min(k, xyz.shape[0])                              # a cube/level can have fewer than k points
    dist = torch.cdist(xyz, xyz)                          # [N, N] pairwise distances (normalised frame)
    return dist.topk(k, dim=1, largest=False).indices    # [N, k] nearest-point indices


class LocalSpatialEncoding(nn.Module):
    """RandLA LocSE: encode each neighbour's relative position and concatenate it with the
    neighbour's (projected) feature, giving one vector per (point, neighbour) pair."""

    def __init__(self, in_channel, out_channel):
        super().__init__()
        # Relative position encoding input is [p_i (3), p_j (3), p_i - p_j (3), ||p_i - p_j|| (1)] = 10.
        self.pos_mlp = nn.Sequential(nn.Linear(10, out_channel), nn.ReLU())
        # Project neighbour features to out_channel so the two concatenated halves are even.
        self.feat_fc = nn.Linear(in_channel, out_channel)

    def forward(self, xyz, neigh_idx, feat):
        # xyz [N, 3]; neigh_idx [N, k]; feat [N, Cin].
        num_points, k = neigh_idx.shape
        neigh_xyz = xyz[neigh_idx]                        # [N, k, 3] neighbour coordinates
        center = xyz.unsqueeze(1).expand(-1, k, -1)       # [N, k, 3] centre point repeated over k
        relative = center - neigh_xyz                     # [N, k, 3] centre minus neighbour
        distance = torch.linalg.norm(relative, dim=2, keepdim=True)   # [N, k, 1] neighbour distance
        # The 10-d relative position fed to the position MLP (RandLA's LocSE encoding).
        pos_input = torch.cat([center, neigh_xyz, relative, distance], dim=2)   # [N, k, 10]
        pos_enc = self.pos_mlp(pos_input)                 # [N, k, out] encoded relative position
        neigh_feat = self.feat_fc(feat)[neigh_idx]        # [N, k, out] projected neighbour features
        return torch.cat([pos_enc, neigh_feat], dim=2)    # [N, k, 2*out]


class AttentivePooling(nn.Module):
    """RandLA attentive pooling: learn softmax weights over the k neighbours and take the
    weighted sum, then an MLP. Replaces a fixed max/avg pool with a learned aggregation."""

    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.score_fc = nn.Linear(in_channel, in_channel)   # per-channel attention logits
        self.mlp = nn.Sequential(nn.Linear(in_channel, out_channel), nn.ReLU())

    def forward(self, neigh_feat):
        # neigh_feat [N, k, Cin]; softmax is over the neighbour dimension k.
        scores = torch.softmax(self.score_fc(neigh_feat), dim=1)   # [N, k, Cin] weights over k
        pooled = (scores * neigh_feat).sum(dim=1)                  # [N, Cin] attentive sum over neighbours
        return self.mlp(pooled)                                    # [N, Cout]


class LocalFeatureAggregation(nn.Module):
    """One RandLA LFA unit with a residual connection: LocSE -> AttentivePooling, added to a
    linear shortcut of the input feature (the dilated-residual idea, kept compact)."""

    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.locse = LocalSpatialEncoding(in_channel, out_channel)
        self.pool = AttentivePooling(2 * out_channel, out_channel)   # LocSE output is 2*out wide
        self.shortcut = nn.Linear(in_channel, out_channel)          # match dims for the residual add
        # track_running_stats=False: this model runs ONE cube at a time (variable-N path) and the
        # encoder downsamples to tiny coarse levels (down to ~16 points), so BatchNorm's running
        # statistics estimated from those few, randomly-sampled points are unreliable and diverge
        # from the batch stats once training overfits -> eval output collapses to one class. Using
        # the cube's OWN stats at both train and eval removes that train/eval mismatch and the
        # collapse. (siamese_kpconv keeps default BN because it runs at native resolution with
        # thousands of points, where running stats are stable.)
        self.bn = nn.BatchNorm1d(out_channel, track_running_stats=False)
        self.relu = nn.ReLU()

    def forward(self, xyz, neigh_idx, feat):
        encoded = self.locse(xyz, neigh_idx, feat)        # [N, k, 2*out]
        pooled = self.pool(encoded)                       # [N, out]
        out = pooled + self.shortcut(feat)                # residual connection
        return self.relu(self.bn(out))                    # [N, out]


def random_sample_indices(num_points, ratio, floor, device):
    """RandLA random sampling for downsampling: keep about num_points/ratio points, never fewer
    than `floor` (so the coarsest level still has enough points for kNN). Returns a LongTensor of
    unique indices on `device`."""
    keep = max(num_points // ratio, min(num_points, floor))   # downsample but respect a floor
    perm = torch.randperm(num_points, device=device)          # random permutation (RandLA's RS)
    return perm[:keep]                                        # [keep] sampled indices


def nearest_upsample(query_xyz, support_xyz, support_feat):
    """Decoder upsampling by nearest neighbour: every query (finer-level) point copies the feature
    of its nearest support (coarser-level) point. Coords are normalised, so cdist is safe."""
    dist = torch.cdist(query_xyz, support_xyz)            # [Nq, Ns] normalised frame
    nn_idx = dist.argmin(dim=1)                           # [Nq] nearest coarse point per query point
    return support_feat[nn_idx]                           # [Nq, C] interpolated features


class RandLAEncoder(nn.Module):
    """Compact RandLA-Net U-Net producing a 128-d per-point feature. Two random-sampling
    downsample stages each with an LFA block, then nearest-neighbour upsampling with skip
    connections. k, the sampling ratio, the floor, and the channel widths are the only
    RandLA hyperparameters."""

    def __init__(self, k=16, ratio=4, floor=16):
        super().__init__()
        self.k = k                                        # neighbours per point in each LocSE
        self.ratio = ratio                                # random-sampling downsample factor per level
        self.floor = floor                                # minimum points to keep at a level
        # Initial per-point feature from the normalised xyz (geometry-only first layer, as in RandLA).
        self.input_fc = nn.Sequential(nn.Linear(3, 32), nn.ReLU())
        # Encoder LFA blocks at levels 0, 1, 2 (channels grow as the point count shrinks).
        self.enc0 = LocalFeatureAggregation(32, 32)
        self.enc1 = LocalFeatureAggregation(32, 64)
        self.enc2 = LocalFeatureAggregation(64, 128)
        # Decoder MLPs after concatenating the upsampled coarse feature with the encoder skip.
        self.dec1 = nn.Sequential(nn.Linear(128 + 64, 64), nn.ReLU())
        self.dec0 = nn.Sequential(nn.Linear(64 + 32, 128), nn.ReLU())

    def forward(self, xyz):
        device = xyz.device
        # ---- Encoder (downsample by random sampling, LFA at each level) ----
        feat0 = self.input_fc(xyz)                        # [N, 32] level-0 features
        idx0 = knn_index(xyz, self.k)                     # level-0 neighbour graph
        feat0 = self.enc0(xyz, idx0, feat0)               # [N, 32]

        sub1 = random_sample_indices(xyz.shape[0], self.ratio, self.floor, device)   # -> level 1
        xyz1 = xyz[sub1]                                  # [N1, 3]
        idx1 = knn_index(xyz1, self.k)
        feat1 = self.enc1(xyz1, idx1, feat0[sub1])        # [N1, 64]

        sub2 = random_sample_indices(xyz1.shape[0], self.ratio, self.floor, device)  # -> level 2
        xyz2 = xyz1[sub2]                                 # [N2, 3]
        idx2 = knn_index(xyz2, self.k)
        feat2 = self.enc2(xyz2, idx2, feat1[sub2])        # [N2, 128] bottleneck features

        # ---- Decoder (nearest-neighbour upsample + encoder skip) ----
        up1 = nearest_upsample(xyz1, xyz2, feat2)         # [N1, 128] bottleneck -> level 1
        dec1 = self.dec1(torch.cat([up1, feat1], dim=1))  # [N1, 64] with the level-1 skip
        up0 = nearest_upsample(xyz, xyz1, dec1)           # [N, 64] level 1 -> level 0
        dec0 = self.dec0(torch.cat([up0, feat0], dim=1))  # [N, 128] with the level-0 skip
        return dec0                                       # [N, 128] per-point feature


def nearest_t0_feature_single(xyz0, xyz1, feat0):
    """Per-cube t1->t0 nearest-feature gather (same mechanism as siamese_kpconv): for each t1
    point return its nearest t0 point's feature in the shared normalised frame. NN is used only
    to pick the correspondence index; the distance value is discarded, so no Chamfer distance
    leaks in (Section 14). xyz0 [N0, 3], xyz1 [N1, 3], feat0 [N0, C] -> matched [N1, C]."""
    dist = torch.cdist(xyz1, xyz0)                        # [N1, N0] normalised frame (safe)
    nn_index = dist.argmin(dim=1)                         # [N1] nearest t0 point per t1 point
    return feat0[nn_index]                                # [N1, C] matched t0 features


class RandLA(nn.Module):
    """Siamese RandLA change head. forward(xyz0, xyz1) -> per-point t1 logits [N1, 2]."""

    def __init__(self, num_classes=2):
        super().__init__()
        self.encoder = RandLAEncoder()
        # Same head as siamese_kpconv: t1 local (128) + change diff (128) + t1 global (128)
        # + t0 global (128) = 512 -> 256 -> 128 -> 2 (per-point Linear layers).
        # track_running_stats=False here too, for the same single-cube reason (consistent
        # train/eval normalisation; the head sees the full t1 cube so stats are well-defined).
        self.seg_mlp = nn.Sequential(
            nn.Linear(512, 256), nn.BatchNorm1d(256, track_running_stats=False), nn.ReLU(),
            nn.Linear(256, 128), nn.BatchNorm1d(128, track_running_stats=False), nn.ReLU(),
            nn.Linear(128, num_classes),                  # final layer -> raw logits, no activation
        )

    def forward(self, xyz0, xyz1):
        # Shared-weight encoder on both cubes (weight tying -> Siamese). One cube at a time.
        feat0 = self.encoder(xyz0)                         # [N0, 128] t0 per-point features
        feat1 = self.encoder(xyz1)                         # [N1, 128] t1 per-point features
        matched_t0 = nearest_t0_feature_single(xyz0, xyz1, feat0)   # [N1, 128]
        change_diff = feat1 - matched_t0                   # [N1, 128] learned per-point change signal
        # Global cube descriptors (max over points), broadcast to every t1 point for context.
        global1 = feat1.max(dim=0).values                  # [128] t1 global
        global0 = feat0.max(dim=0).values                  # [128] t0 global
        num_points = xyz1.shape[0]
        g1 = global1.unsqueeze(0).expand(num_points, -1)   # [N1, 128]
        g0 = global0.unsqueeze(0).expand(num_points, -1)   # [N1, 128]
        seg_in = torch.cat([feat1, change_diff, g1, g0], dim=1)   # [N1, 512]
        return self.seg_mlp(seg_in)                        # [N1, 2] per-point logits
