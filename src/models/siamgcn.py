"""
Siamese Graph CNN for per-point change detection on a bi-temporal cube pair
(Section 2 model `siamgcn`, a fixed-N model).

Same proven design as the Task 5 template (siamese_pointnet): a SHARED encoder embeds
both cubes, then every t1 point gathers its nearest t0 point's LEARNED feature in the
shared normalised frame and the head sees the difference (feat_t1 - matched_t0). Only
the encoder differs: a DGCNN / EdgeConv graph encoder (Wang et al.) that builds a
DYNAMIC k-nearest-neighbour graph and convolves over edge features, so a point's
representation is shaped by its local neighbourhood and re-computed in feature space at
each layer. The NN feature-diff head (nearest_t0_features) is reused verbatim, because
that local t0->t1 correspondence is what actually localises change.

Self-contained pure torch, kNN graph (no CUDA ops). Fixed-N padding is handled in the
graph: padded slots are never chosen as neighbours (their distance is pushed to a huge
value before the topk) and are excluded from the global max-pool. Padded query points
still get features (from real neighbours), but the loss mask discards them. All kNN /
cdist run on normalised coords or learned features (never million-scale raw coords), so
the Task 5 float32-cancellation bug cannot occur here.

Uniform fixed-N forward contract (identical to siamese_pointnet):
    forward(xyz0, xyz1, mask0, mask1) -> per-point t1 logits [B, N, 2]
"""

import torch
import torch.nn as nn

# Reuse the proven t1->t0 nearest-neighbour feature-difference helper (Task 5 mechanism).
from .siamese_pointnet import nearest_t0_features


def index_points(points, idx):
    """Gather [B, N, C] by idx [B, N, K] -> [B, N, K, C] (advanced indexing)."""
    batch_size = points.shape[0]
    index_shape = idx.shape[1:]                           # (N, K)
    batch_index = torch.arange(batch_size, device=points.device)
    batch_index = batch_index.view(batch_size, *([1] * len(index_shape)))
    batch_index = batch_index.expand(batch_size, *index_shape)
    return points[batch_index, idx]


def knn_graph(feature, mask, k):
    """
    Build a k-nearest-neighbour graph over REAL points. feature [B, C, N] is the space
    the graph is built in (raw xyz for the first layer, learned features afterwards).
    Returns neighbour indices [B, N, k]. Padded points are pushed to a huge distance so
    they are never selected as neighbours of any point.
    """
    feature_t = feature.transpose(1, 2)                   # [B, N, C]
    dist = torch.cdist(feature_t, feature_t)              # [B, N, N] pairwise distance
    dist = dist.masked_fill(~mask.unsqueeze(1), 1e10)     # padded columns -> never a neighbour
    return dist.topk(k, dim=2, largest=False).indices     # [B, N, k] nearest real points


def edge_feature(feature, neighbour_idx):
    """
    Build EdgeConv input: for each point i and neighbour j, concat (x_j - x_i, x_i).
    feature [B, C, N]; neighbour_idx [B, N, k] -> [B, 2C, N, k].
    """
    feature_t = feature.transpose(1, 2)                   # [B, N, C]
    k = neighbour_idx.shape[2]
    neighbours = index_points(feature_t, neighbour_idx)   # [B, N, k, C]
    centre = feature_t.unsqueeze(2).expand(-1, -1, k, -1)  # [B, N, k, C]
    edge = torch.cat([neighbours - centre, centre], dim=3)  # [B, N, k, 2C]
    return edge.permute(0, 3, 1, 2)                        # [B, 2C, N, k]


class EdgeConv(nn.Module):
    """One EdgeConv block: shared MLP over edge features, then max over the k neighbours."""

    def __init__(self, in_channel, out_channel):
        super().__init__()
        # Input is 2*in_channel because edge_feature concatenates (x_j - x_i, x_i).
        self.conv = nn.Sequential(
            nn.Conv2d(2 * in_channel, out_channel, 1), nn.BatchNorm2d(out_channel), nn.ReLU(),
        )

    def forward(self, feature, neighbour_idx):
        edge = edge_feature(feature, neighbour_idx)       # [B, 2C, N, k]
        return self.conv(edge).max(dim=3).values          # [B, out, N] pool over neighbours


class SharedDGCNNEncoder(nn.Module):
    """
    Shared DGCNN encoder: three EdgeConv layers on a dynamic kNN graph (rebuilt in
    feature space each layer), concatenated into a per-point feature, projected to a
    128-d local feature for matching, plus a 1024-d global descriptor (masked max-pool).
    Returns (point_feat [B, 128, N], global_feat [B, 1024]).
    """

    def __init__(self, k=20):
        super().__init__()
        self.k = k
        self.edge_conv1 = EdgeConv(3, 64)
        self.edge_conv2 = EdgeConv(64, 64)
        self.edge_conv3 = EdgeConv(64, 128)
        # Project the concatenated multi-scale features to the per-point matching feature.
        self.point_proj = nn.Sequential(
            nn.Conv1d(64 + 64 + 128, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
        )
        # Lift the same concat to a wide feature before the global max-pool.
        self.global_proj = nn.Sequential(
            nn.Conv1d(64 + 64 + 128, 1024, 1), nn.BatchNorm1d(1024), nn.ReLU(),
        )

    def forward(self, xyz, mask):
        x = xyz.transpose(1, 2)                            # [B, 3, N] channels-first input
        idx = knn_graph(x, mask, self.k)                   # first graph in coordinate space
        f1 = self.edge_conv1(x, idx)                       # [B, 64, N]
        idx = knn_graph(f1, mask, self.k)                  # dynamic graph in feature space
        f2 = self.edge_conv2(f1, idx)                      # [B, 64, N]
        idx = knn_graph(f2, mask, self.k)
        f3 = self.edge_conv3(f2, idx)                      # [B, 128, N]
        multi_scale = torch.cat([f1, f2, f3], dim=1)       # [B, 256, N]
        point_feat = self.point_proj(multi_scale)          # [B, 128, N] per-point matching feature
        lifted = self.global_proj(multi_scale)             # [B, 1024, N]
        # Masked max-pool: padded slots set very negative so they never win the global.
        very_negative = torch.finfo(lifted.dtype).min
        masked = lifted.masked_fill(~mask.unsqueeze(1), very_negative)
        global_feat = masked.max(dim=2).values             # [B, 1024] global cube descriptor
        return point_feat, global_feat


class SiamGCN(nn.Module):
    """Siamese DGCNN change head. forward -> per-point t1 logits [B, N, 2]."""

    def __init__(self, num_classes=2):
        super().__init__()
        self.encoder = SharedDGCNNEncoder()
        # Seg head input per t1 point = t1 local (128) + change diff (128)
        # + t1 global (1024) + t0 global (1024) = 2304 channels (same layout as siamese_pointnet).
        self.seg_mlp = nn.Sequential(
            nn.Conv1d(2304, 512, 1), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Conv1d(512, 256, 1), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Conv1d(256, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, num_classes, 1),                # final layer -> raw logits, no activation
        )

    def forward(self, xyz0, xyz1, mask0, mask1):
        # Shared encoder on both cubes (weights tied -> Siamese).
        feat0, global0 = self.encoder(xyz0, mask0)         # t0 per-point features + global
        feat1, global1 = self.encoder(xyz1, mask1)         # t1 per-point features + global
        # Match each t1 point to its nearest t0 point and take the learned difference.
        matched_t0 = nearest_t0_features(xyz0, xyz1, feat0, mask0)   # [B, 128, N]
        change_diff = feat1 - matched_t0                   # [B, 128, N] learned per-point change signal
        num_points = xyz1.shape[1]
        g1 = global1.unsqueeze(2).expand(-1, -1, num_points)   # [B, 1024, N] t1 global
        g0 = global0.unsqueeze(2).expand(-1, -1, num_points)   # [B, 1024, N] t0 global
        seg_in = torch.cat([feat1, change_diff, g1, g0], dim=1)  # [B, 2304, N]
        logits = self.seg_mlp(seg_in)                            # [B, 2, N]
        return logits.transpose(1, 2)                            # [B, N, 2] back to points-first
