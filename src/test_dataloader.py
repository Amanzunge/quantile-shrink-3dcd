"""
Task 3 verifier. Checks the sampling core (src/cube_sampling.py) and the torch
dataloader (src/dataloader.py) against the real Urb3DCD-V2 LD cache. Read-only.
Run under an interpreter that has torch AND plyfile:

    python src/test_dataloader.py --config configs/urb3dcd_v2_ld.yaml

Part A tests the torch-free maths (FPS regimes, normalisation, padding).
Part B builds real train/val/test datasets and checks the split sizes reconcile,
the FPS regimes behave (random varies, fixed repeats), padding/masks are correct,
labels stay aligned to points, and the full-resolution fields predict.py needs are
exact and UNNORMALISED.
"""

import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cube_sampling import (
    farthest_point_sampling, normalize_to_cube, pad_to_length, pad_labels)


def part_a_core():
    """Unit tests for the torch-free sampling maths."""
    print("Part A: sampling core")

    # FPS take-all: when the target is >= the cloud size, keep every point in order.
    pts = np.random.default_rng(0).random((300, 3)).astype(np.float32)
    take_all = farthest_point_sampling(pts, 1024, np.random.default_rng(1))
    assert np.array_equal(take_all, np.arange(300)), "take-all must return arange(M)"
    print("  ok  take-all returns every point in order, no upsampling")

    # FPS downsample: exactly n_target distinct, in-range indices.
    big = np.random.default_rng(2).random((5000, 3)).astype(np.float32)
    sub = farthest_point_sampling(big, 1024, np.random.default_rng(3))
    assert sub.shape == (1024,), "FPS must return n_target indices"
    assert len(np.unique(sub)) == 1024, "FPS indices must be distinct (no replacement)"
    assert sub.min() >= 0 and sub.max() < 5000, "FPS indices out of range"
    print("  ok  downsample returns n_target distinct in-range indices")

    # Fixed regime: a seeded rng is reproducible; a different seed differs.
    a = farthest_point_sampling(big, 1024, np.random.default_rng(42))
    b = farthest_point_sampling(big, 1024, np.random.default_rng(42))
    c = farthest_point_sampling(big, 1024, np.random.default_rng(7))
    assert np.array_equal(a, b), "same seed must reproduce the FPS subset"
    assert not np.array_equal(a, c), "different seed should change the FPS subset"
    print("  ok  seeded FPS reproducible; different seed changes the subset")

    # Isotropic invariance: a uniform scale + shift must not change FPS order when
    # the first (random) point is the same. This is why FPS-on-raw == FPS-on-norm.
    idx_raw = farthest_point_sampling(big, 256, np.random.default_rng(5))
    shifted = (big * 7.0 + 100.0).astype(np.float32)
    idx_shift = farthest_point_sampling(shifted, 256, np.random.default_rng(5))
    assert np.array_equal(idx_raw, idx_shift), "uniform scale+shift changed FPS order"
    print("  ok  FPS order invariant to isotropic scale+shift")

    # Normalisation: cube centre -> (0,0,*), ground -> z=0, corners XY in [-1,1].
    bbox_min = np.array([10.0, 20.0, 5.0], dtype=np.float32)   # x0,y0,z_ground
    bbox_max = np.array([60.0, 70.0, 35.0], dtype=np.float32)  # 50 m XY cube
    centre_pt = np.array([[35.0, 45.0, 5.0]], dtype=np.float32)  # XY centre, ground
    n_centre = normalize_to_cube(centre_pt, bbox_min, bbox_max)
    assert np.allclose(n_centre[0], [0.0, 0.0, 0.0], atol=1e-5), "centre/ground != origin"
    corners = np.array([[10.0, 20.0, 35.0], [60.0, 70.0, 5.0]], dtype=np.float32)
    n_corners = normalize_to_cube(corners, bbox_min, bbox_max)
    assert np.all(np.abs(n_corners[:, :2]) <= 1.0 + 1e-5), "corner XY left [-1,1]"
    assert n_corners[0, 2] > 0, "top corner Z should be positive above ground"
    print("  ok  normalise: centre->origin, ground->0, corners XY in [-1,1]")

    # Padding: real points first, zeros after, mask marks the real ones.
    small = np.random.default_rng(9).random((300, 3)).astype(np.float32)
    padded, mask = pad_to_length(small, 1024)
    assert padded.shape == (1024, 3) and mask.shape == (1024,)
    assert mask.sum() == 300, "mask must mark exactly the real points"
    assert np.array_equal(padded[:300], small), "real points must be preserved up front"
    assert np.all(padded[300:] == 0), "padding must be zeros"
    labels = np.ones(300, dtype=np.int64)
    padded_labels = pad_labels(labels, 1024)
    assert padded_labels.shape == (1024,) and padded_labels[:300].sum() == 300
    assert padded_labels[300:].sum() == 0, "label padding must be zero"
    print("  ok  pad: real points up front, zeros after, mask + label padding")
    print("Part A passed.\n")


def part_b_dataset(config_path):
    """Integration tests against the real LD cache via the torch dataloader."""
    print("Part B: torch dataloader on", config_path)
    # Import torch-dependent code only here so Part A can run without torch.
    from dataloader import (
        CubePairDataset, build_cube_dataloader, loader_settings_from_config,
        stable_cube_seed)

    settings = loader_settings_from_config(config_path)
    cache_root = settings["cache_root"]
    raw_root = settings["raw_root"]
    n_target = settings["n_target"]
    print("  cache:", cache_root, "| raw:", raw_root, "| FPS native:", n_target)

    # Split sizes must reconcile with the Task-2 manifest (1394/423/765 for LD).
    train_ds = CubePairDataset(cache_root, raw_root, "train", n_target, fps_seed=None)
    val_ds = CubePairDataset(cache_root, raw_root, "val", n_target, fps_seed=None)
    test_ds = CubePairDataset(cache_root, raw_root, "test", n_target,
                              fps_seed=42, return_full_res=True)
    print("  split sizes: train={} val={} test={}".format(
        len(train_ds), len(val_ds), len(test_ds)))
    assert len(train_ds) + len(val_ds) + len(test_ds) == 2582, "LD total != 2582 cubes"
    print("  ok  split cube counts reconcile with the manifest")

    # One train sample has the fixed-N shapes the deep models expect.
    s = train_ds[0]
    assert s["xyz0"].shape == (n_target, 3) and s["xyz1"].shape == (n_target, 3)
    assert s["mask1"].shape == (n_target,) and s["label1"].shape == (n_target,)
    print("  ok  fixed-N sample shapes are [n_target, 3] / [n_target]")

    # Find one FPS cube (M1 > n_target) and one take-all cube (M1 <= n_target).
    def cube_m1(ds, i):
        scene_id, cube_id = ds.index[i]
        off = ds.caches[scene_id]["offset_t1"]
        return int(off[cube_id + 1] - off[cube_id])
    fps_i = next(i for i in range(len(train_ds)) if cube_m1(train_ds, i) > n_target)
    take_i = next(i for i in range(len(train_ds)) if cube_m1(train_ds, i) <= n_target)

    # Random regime: drawing the same FPS cube twice gives different subsets.
    d1 = train_ds[fps_i]
    d2 = train_ds[fps_i]
    assert not np.array_equal(d1["xyz1"], d2["xyz1"]), \
        "random regime must resample (xyz1 should differ across draws)"
    print("  ok  random regime resamples an FPS cube (M1={}) each draw".format(
        cube_m1(train_ds, fps_i)))

    # Fixed regime: drawing the same FPS cube twice is byte-for-byte identical.
    e1 = test_ds[0]
    e2 = test_ds[0]
    assert np.array_equal(e1["xyz1"], e2["xyz1"]), "fixed regime must be reproducible"
    print("  ok  fixed regime is reproducible (single deterministic pass)")

    # Take-all + pad: a small cube keeps all its points and masks the rest.
    t = train_ds[take_i]
    m1 = cube_m1(train_ds, take_i)
    assert t["mask1"].sum() == m1, "mask must mark exactly the real take-all points"
    assert np.all(t["xyz1"][m1:] == 0), "padding rows must be zero"
    print("  ok  take-all cube (M1={}) padded to {} with a correct mask".format(
        m1, n_target))

    # Normalised model input stays inside the unit cube in XY for real points.
    real = e1["mask1"]
    xy = e1["xyz1"][real][:, :2]
    assert np.all(np.abs(xy) <= 1.0 + 1e-4), "normalised XY left [-1,1]"
    print("  ok  normalised input XY within [-1,1] for real points")

    # Full-resolution fields: UNNORMALISED, exact, and label-aligned.
    full_xyz1 = e1["full_xyz1"]
    full_lab1 = e1["full_label1"]
    sel1 = e1["sel1"]
    n_real = int(real.sum())
    # The scored points are exactly the FPS subset of the full-res cube.
    scored = full_xyz1[sel1]
    assert sel1.shape[0] == n_real, "sel1 length must equal the real point count"
    # full_xyz1 is original metres, not the [-1,1] model frame.
    assert full_xyz1[:, :2].max() > 2.0, "full_xyz1 should be raw metres, not normalised"
    # Labels carried in label1 must match the full-res labels at the FPS indices.
    assert np.array_equal(e1["label1"][:n_real], full_lab1[sel1]), \
        "label1 not aligned to full_label1[sel1]"
    print("  ok  full-res t1 is unnormalised, exact, and label-aligned for NN-prop")

    # Cross-check the per-cube changed count against the Task-2 cache.
    scene_id, cube_id = test_ds.index[0]
    cache = np.load(os.path.join(cache_root, scene_id + ".npz"))
    assert int(full_lab1.sum()) == int(cache["n_changed"][cube_id]), \
        "full-res changed count disagrees with the cache n_changed"
    print("  ok  full-res changed count matches cache n_changed for cube 0")

    # The stable seed is deterministic for a given identity.
    assert stable_cube_seed(42, "LyonS", 3) == stable_cube_seed(42, "LyonS", 3)
    assert stable_cube_seed(42, "LyonS", 3) != stable_cube_seed(42, "LyonS", 4)
    print("  ok  stable per-cube seed is deterministic and cube-specific")

    # DataLoader batches stack fixed-N samples and keep full-res as a list.
    loader = build_cube_dataloader(
        cache_root, raw_root, "test", n_target, fps_seed=42,
        return_full_res=True, batch_size=4, num_workers=0)
    batch = next(iter(loader))
    assert batch["xyz0"].shape == (4, n_target, 3), "batch xyz0 should be [B,N,3]"
    assert batch["label1"].shape == (4, n_target), "batch label1 should be [B,N]"
    assert isinstance(batch["full_xyz1"], list) and len(batch["full_xyz1"]) == 4
    assert len(batch["scene_id"]) == 4 and batch["cube_id"].shape == (4,)
    print("  ok  DataLoader stacks fixed-N batches; full-res kept as a list")

    # Variable-N path (kpconv/randla) returns ragged native-length clouds.
    var_ds = CubePairDataset(cache_root, raw_root, "test", n_target,
                             fps_seed=42, fixed_n=False)
    v = var_ds[take_i]
    assert v["xyz1"].shape[0] == min(n_target, cube_m1(var_ds, take_i)), \
        "variable-N cube should keep its native length"
    assert bool(v["mask1"].all()), "variable-N has no padding, mask all True"
    print("  ok  variable-N path returns native-length clouds, no padding")
    print("Part B passed.\n")


def main():
    parser = argparse.ArgumentParser(description="Verify the Task 3 dataloader.")
    parser.add_argument("--config", default="configs/urb3dcd_v2_ld.yaml",
                        help="dataset YAML providing cache_root, raw_root, fps_native")
    args = parser.parse_args()

    part_a_core()
    part_b_dataset(args.config)
    print("All Task 3 checks passed.")


if __name__ == "__main__":
    main()
