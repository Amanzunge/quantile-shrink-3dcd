"""
Siamese PointNet for per-point change detection on a bi-temporal cube pair
(Section 2 model `siamese_pointnet`, a fixed-N model).

Design (revised after the first LD run showed the global-only head could not
LOCALISE change: AUC ~0.6, per-cube oracle F1 ~0.3, below the ICP baseline):

  - One PointNet encoder with SHARED weights embeds both cubes (this weight tying
    is what makes it Siamese). Input is normalised XYZ only, 3 channels; no T-Net
    (the dataloader already places points in the canonical cube frame).
  - The encoder returns a per-point feature [B,128,N] and one global cube descriptor
    [B,1024] (masked max-pool over REAL points only).
  - LOCAL CORRESPONDENCE (the key change): for every t1 point we find its nearest t0
    point in the SHARED normalised cube frame and gather that t0 point's LEARNED
    feature. Because the normalisation is shared and isotropic, an unchanged t1 point
    sits on top of its t0 twin (feature difference ~ 0), while a changed t1 point has
    no good t0 match (large feature difference). The head therefore sees a learned,
    per-point change signal (feat_t1 - matched_feat_t0), not just a global summary.
    This is the Siamese-KPConv-style mechanism that is standard/SOTA on Urb3DCD.
  - NN is used ONLY to pick the correspondence index; the geometric distance VALUE is
    discarded. The change signal is the learned feature difference, so this stays
    within "only model output scores and logits, no Chamfer distance" (Section 14).
  - The change head concatenates, per t1 point: t1 local feature, the t1-minus-matched
    feature difference, the t1 global, and the t0 global -> segmentation MLP -> change
    logit.

Uniform forward contract for ALL fixed-N models (so train.py / predict.py call every
model the same way):
    forward(xyz0, xyz1, mask0, mask1) -> per-point t1 logits [B, N, 2]
where xyz* are [B, N, 3] normalised input and mask* are [B, N] bool (True = real).

BatchNorm sees the padded (zero) input slots during training; they are a minority
(only take-all cubes are padded) and consistent train/test, so the effect on the
running statistics is negligible. The masked placements that matter are correct: the
global max-pool excludes padded points, the NN match excludes padded t0 candidates,
and the loss excludes padded t1 points. At inference (eval mode, frozen BN stats) a
real point's logit does not depend on any padded slot.
"""

import torch
import torch.nn as nn


class SharedPointNetEncoder(nn.Module):
    """
    PointNet encoder applied (with shared weights) to each cube of the pair.
    Maps [B, 3, N] -> (per-point features [B, 128, N], global descriptor [B, 1024]).
    The per-point features are what the change head matches across the two clouds.
    """

    def __init__(self):
        super().__init__()
        # Shared per-point MLP -> 128-d feature used for matching AND for the global.
        # Conv1d with kernel 1 is the standard PointNet shared per-point MLP.
        self.point_mlp = nn.Sequential(
            nn.Conv1d(3, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
        )
        # Lift the per-point feature before the global max-pool.
        self.global_mlp = nn.Sequential(
            nn.Conv1d(128, 256, 1), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Conv1d(256, 1024, 1), nn.BatchNorm1d(1024), nn.ReLU(),
        )

    def forward(self, xyz, mask):
        # xyz [B, 3, N] model input; mask [B, N] bool, True for real points.
        point_feat = self.point_mlp(xyz)            # [B, 128, N] per-point feature
        pre_pool = self.global_mlp(point_feat)      # [B, 1024, N] pre-pool features
        # Masked max-pool: padded slots set to a large negative so they never win.
        very_negative = torch.finfo(pre_pool.dtype).min
        masked = pre_pool.masked_fill(~mask.unsqueeze(1), very_negative)
        global_feat = masked.max(dim=2).values      # [B, 1024] global cube descriptor
        return point_feat, global_feat


def nearest_t0_features(xyz0, xyz1, feat0, mask0):
    """
    For every t1 point, gather the LEARNED feature of its nearest t0 point in the
    shared normalised cube frame. NN is used only to choose the correspondence index;
    the distance value is discarded (the change signal is the learned feature diff).

    xyz0, xyz1 [B, N, 3] normalised coords; feat0 [B, C, N0] t0 per-point features;
    mask0 [B, N0] bool (True = real t0 point). Returns matched_t0 [B, C, N1].
    """
    # Pairwise t1 -> t0 distances in the shared normalised frame.
    dist = torch.cdist(xyz1, xyz0)                  # [B, N1, N0]
    # Padded t0 points must never be chosen as a match: push their distance to +max.
    very_big = torch.finfo(dist.dtype).max
    dist = dist.masked_fill(~mask0.unsqueeze(1), very_big)
    nn_index = dist.argmin(dim=2)                   # [B, N1] nearest real t0 index per t1 point
    channels = feat0.shape[1]
    # Gather the matched t0 feature for each t1 point along the point dimension.
    gather_index = nn_index.unsqueeze(1).expand(-1, channels, -1)   # [B, C, N1]
    matched = torch.gather(feat0, 2, gather_index)  # [B, C, N1]
    return matched


class SiamesePointNet(nn.Module):
    """Siamese PointNet change head. forward -> per-point t1 logits [B, N, 2]."""

    def __init__(self, num_classes=2):
        super().__init__()
        self.encoder = SharedPointNetEncoder()
        # Seg head input per t1 point = t1 local (128) + change diff (128)
        # + t1 global (1024) + t0 global (1024) = 2304 channels.
        self.seg_mlp = nn.Sequential(
            nn.Conv1d(2304, 512, 1), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Conv1d(512, 256, 1), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Conv1d(256, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, num_classes, 1),          # final layer -> raw logits, no activation
        )

    def forward(self, xyz0, xyz1, mask0, mask1):
        # Dataloader hands [B, N, 3]; PointNet 1x1 convs want channels-first [B, 3, N].
        x0 = xyz0.transpose(1, 2)                    # [B, 3, N] t0 input
        x1 = xyz1.transpose(1, 2)                    # [B, 3, N] t1 input
        # Shared encoder on both cubes (weights tied -> Siamese).
        feat0, global0 = self.encoder(x0, mask0)     # t0 per-point features + global
        feat1, global1 = self.encoder(x1, mask1)     # t1 per-point features + global
        # Match each t1 point to its nearest t0 point and take the learned difference.
        matched_t0 = nearest_t0_features(xyz0, xyz1, feat0, mask0)  # [B, 128, N]
        change_diff = feat1 - matched_t0             # [B, 128, N] learned per-point change signal
        num_points = x1.shape[2]
        # Broadcast both global descriptors to every t1 point for cube-level context.
        g1 = global1.unsqueeze(2).expand(-1, -1, num_points)   # [B, 1024, N] t1 global
        g0 = global0.unsqueeze(2).expand(-1, -1, num_points)   # [B, 1024, N] t0 global
        # Concatenate local t1 + change diff + global t1 + global t0 along channels.
        seg_in = torch.cat([feat1, change_diff, g1, g0], dim=1)  # [B, 2304, N]
        logits = self.seg_mlp(seg_in)                            # [B, 2, N] per-point logits
        return logits.transpose(1, 2)                            # [B, N, 2] back to points-first
