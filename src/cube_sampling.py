"""
Task 3 sampling core: the FPS, normalisation, and padding maths for one cube,
written in plain numpy with NO torch import. Keeping this torch-free means it can
be unit-tested in the CPU-only project env (threshold_env), while the thin torch
Dataset that calls it (src/dataloader.py) is only needed where torch is present
(Kaggle, or the global interpreter for a local smoke test).

Everything here operates on a single cube cloud, [M, 3] float32 raw XYZ. The three
helpers are:
  farthest_point_sampling  pick which points survive (downsample-or-take-all)
  normalize_to_cube        shared isotropic transform into the model input frame
  pad_to_length            grow a take-all cube up to the fixed-N model size + mask

FPS policy (frozen in Task 2): downsample-or-take-all, NEVER upsample, never
sample with replacement. A cube with more points than the target is thinned by
FPS; a cube with fewer is taken whole and (for fixed-N models) padded with a mask.
"""

import numpy as np


def farthest_point_sampling(xyz, n_target, rng):
    """
    Return LOCAL indices (into xyz) of an FPS subset of size min(n_target, M).

    If the cube already has at most n_target points we take them all in order
    (the never-upsample rule); otherwise we run iterative farthest point sampling
    seeded by a random first point drawn from rng, so a fresh rng gives a fresh
    subset (this is the "train FPS random per batch" requirement) and a seeded rng
    gives a reproducible subset (the "test FPS fixed seed" requirement).
    """
    num_points = xyz.shape[0]
    # Never upsample: too-small (or exact) cube keeps every point, original order.
    if n_target >= num_points:
        return np.arange(num_points, dtype=np.int64)

    selected = np.empty(n_target, dtype=np.int64)        # indices we keep
    # Squared distance from every point to the nearest already-selected point.
    nearest_sq = np.full(num_points, np.inf, dtype=np.float64)

    # First seed point is random; its index is what makes the regime random/fixed.
    current = int(rng.integers(0, num_points))
    selected[0] = current
    for i in range(1, n_target):
        # Squared distance from the just-picked point to all points.
        diff = xyz - xyz[current]
        dist_sq = np.einsum("ij,ij->i", diff, diff)      # row-wise squared norm
        # Each point tracks distance to the CLOSEST selected point so far.
        nearest_sq = np.minimum(nearest_sq, dist_sq)
        # The next pick is the point farthest from the current selection.
        current = int(np.argmax(nearest_sq))
        selected[i] = current
    return selected


def normalize_to_cube(xyz, cube_bbox_min, cube_bbox_max):
    """
    Map raw metric XYZ into the model input frame with a SHARED ISOTROPIC
    transform: subtract the cube centre in XY and the cube ground (bbox min Z),
    then divide by half the cube XY extent. The same (centre, scale) is used for
    both clouds of the pair, so the t0/t1 registration is preserved; the scale is
    one scalar (isotropic), so shape is not distorted and FPS order is unchanged.

    XY lands in roughly [-1, 1]; Z starts at 0 at the ground and keeps its true
    proportion to XY. Returns float32 [M, 3]; coords passed in stay untouched.
    """
    # Translation: cube centre in XY, ground level (lowest Z) in Z.
    centre = np.array([
        (cube_bbox_min[0] + cube_bbox_max[0]) * 0.5,
        (cube_bbox_min[1] + cube_bbox_max[1]) * 0.5,
        cube_bbox_min[2],
    ], dtype=np.float32)
    # One isotropic scale: half the cube XY extent (25 m for a 50 m cube).
    half_extent = (cube_bbox_max[0] - cube_bbox_min[0]) * 0.5
    normalized = (xyz - centre) / half_extent
    return normalized.astype(np.float32)


def pad_to_length(xyz, n_target):
    """
    Pad a take-all cube up to the fixed-N model size and return (padded, mask).

    padded is [n_target, 3] float32 with the real points first and zeros after;
    mask is [n_target] bool, True for real points. Padded slots are placeholders
    only and must be excluded from the loss and from the two-moment statistics.
    Input is assumed to hold at most n_target points (FPS already enforced that).
    """
    num_points = xyz.shape[0]
    padded = np.zeros((n_target, 3), dtype=np.float32)
    padded[:num_points] = xyz                            # real points up front
    mask = np.zeros(n_target, dtype=bool)
    mask[:num_points] = True                             # mark the real points
    return padded, mask


def pad_labels(labels, n_target):
    """
    Pad a [M] int64 label vector up to n_target with zeros (the unchanged class).
    The padding value is irrelevant because the mask removes padded slots before
    they reach the loss; zero is just a safe, in-range filler.
    """
    num_points = labels.shape[0]
    padded = np.zeros(n_target, dtype=np.int64)
    padded[:num_points] = labels
    return padded
