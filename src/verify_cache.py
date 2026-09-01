"""
Verifier for the Task 2 cube caches. Loads every per-scene .npz listed in a
dataset's manifest.json and asserts the structural invariants the rest of the
pipeline relies on. Read-only; it never writes. Run it after preprocess_cubes.py
and reuse it unchanged for HKCD and IndoorCD.

Run:
    python src/verify_cache.py --cache-root data/cache/urb3dcd_v2_ld
    python src/verify_cache.py --cache-root data/cache/urb3dcd_v2_ms
"""

import os
import json
import argparse
import numpy as np

# Keys every cache file must carry (scene metadata + per-cube + CSR index arrays).
REQUIRED_KEYS = [
    "dataset", "scene_id", "split", "cube_size_xy", "min_points",
    "num_source_classes", "n_points_t0", "n_points_t1",
    "scene_bbox_min", "scene_bbox_max", "grid_origin_xy",
    "cube_id", "cube_grid_ix", "cube_grid_iy", "cube_bbox_min", "cube_bbox_max",
    "count_t0", "count_t1", "n_unchanged", "n_changed", "class_hist",
    "idx_t0", "offset_t0", "idx_t1", "offset_t1",
]


def check_csr(indices, offset, counts, num_points, min_points, cloud_name):
    """Assert the CSR layout of one cloud is self-consistent and in range."""
    num_cubes = len(counts)
    # Offset array has one slot per cube plus a closing total.
    assert offset.shape[0] == num_cubes + 1, cloud_name + " offset length"
    assert offset[0] == 0, cloud_name + " offset must start at 0"
    # Offsets are non-decreasing and end at the concatenated index length.
    assert np.all(np.diff(offset) >= 0), cloud_name + " offset not monotonic"
    assert offset[-1] == indices.shape[0], cloud_name + " offset end != idx len"
    # Each cube's slice length equals its stored point count.
    slice_lengths = np.diff(offset)
    assert np.array_equal(slice_lengths, counts.astype(np.int64)), \
        cloud_name + " slice lengths != counts"
    # Drop rule: every kept cube has at least min_points in this cloud.
    assert np.all(counts >= min_points), cloud_name + " count below min_points"
    # Indices address real points and never repeat (each point in one cube max).
    if indices.shape[0] > 0:
        assert indices.min() >= 0 and indices.max() < num_points, \
            cloud_name + " index out of range"
        assert len(np.unique(indices)) == indices.shape[0], \
            cloud_name + " duplicate point indices"


def verify_scene(npz_path):
    """Verify one scene cache file. Returns its kept-cube count and split."""
    cache = np.load(npz_path, allow_pickle=False)

    # Every required key must be present.
    for key in REQUIRED_KEYS:
        assert key in cache.files, "missing key " + key + " in " + npz_path

    min_points = int(cache["min_points"])
    num_classes = int(cache["num_source_classes"])
    n_points_t0 = int(cache["n_points_t0"])
    n_points_t1 = int(cache["n_points_t1"])

    cube_id = cache["cube_id"]
    num_cubes = len(cube_id)
    # cube_id is the canonical 0..K-1 range used as the downstream index.
    assert np.array_equal(cube_id, np.arange(num_cubes, dtype=np.int32)), \
        "cube_id not 0..K-1"

    count_t0 = cache["count_t0"]
    count_t1 = cache["count_t1"]

    # CSR consistency and index validity for both clouds.
    check_csr(cache["idx_t0"], cache["offset_t0"], count_t0, n_points_t0,
              min_points, "t0")
    check_csr(cache["idx_t1"], cache["offset_t1"], count_t1, n_points_t1,
              min_points, "t1")

    # Label closure: binary split and class histogram both account for every
    # t1 point in the cube, and the changed count is classes 1..6.
    n_unchanged = cache["n_unchanged"]
    n_changed = cache["n_changed"]
    class_hist = cache["class_hist"]
    assert np.array_equal(n_unchanged + n_changed, count_t1), \
        "unchanged + changed != count_t1"
    assert np.array_equal(class_hist.sum(axis=1), count_t1.astype(np.int64)), \
        "class_hist rows != count_t1"
    assert np.array_equal(class_hist[:, 1:].sum(axis=1), n_changed.astype(np.int64)), \
        "class_hist changed columns != n_changed"
    assert class_hist.shape[1] == num_classes, "class_hist width != num_classes"

    # Spatial bbox sanity: XY strictly grows by the cube size; Z is shared.
    bbox_min = cache["cube_bbox_min"]
    bbox_max = cache["cube_bbox_max"]
    assert np.all(bbox_max[:, 0] > bbox_min[:, 0]), "bbox X not increasing"
    assert np.all(bbox_max[:, 1] > bbox_min[:, 1]), "bbox Y not increasing"
    assert np.all(bbox_max[:, 2] >= bbox_min[:, 2]), "bbox Z inverted"

    split = str(cache["split"])
    return num_cubes, split


def main():
    parser = argparse.ArgumentParser(description="Verify Task 2 cube caches.")
    parser.add_argument("--cache-root", required=True, help="dir with the .npz caches")
    args = parser.parse_args()

    manifest_file = os.path.join(args.cache_root, "manifest.json")
    with open(manifest_file) as f:
        manifest = json.load(f)

    print("Verifying cache root:", args.cache_root)
    print("Dataset:", manifest["dataset"], "| scenes:", len(manifest["scenes"]))
    print("")

    total_cubes = 0
    total_bytes = 0
    by_split = {"train": 0, "val": 0, "test": 0}
    for scene in manifest["scenes"]:
        npz_path = os.path.join(args.cache_root, scene["scene_id"] + ".npz")
        num_cubes, split = verify_scene(npz_path)
        # The cache must agree with the manifest it was written alongside.
        assert num_cubes == scene["cubes_kept"], \
            "manifest/cache cube count mismatch for " + scene["scene_id"]
        by_split[split] = by_split.get(split, 0) + num_cubes
        total_cubes += num_cubes
        total_bytes += os.path.getsize(npz_path)
        print("  OK {:<10} cubes={:>4d} ({})".format(
            scene["scene_id"][:10], num_cubes, split))

    # The dataset-level total must match the manifest's recorded total.
    assert total_cubes == manifest["total_cubes_kept"], "total cube mismatch"

    print("")
    print("All scenes verified.")
    print("Kept cubes by split:", "train", by_split["train"],
          "val", by_split["val"], "test", by_split["test"])
    print("Total kept cubes:", total_cubes,
          "(manifest:", manifest["total_cubes_kept"], ")")
    print("Cache size on disk: {:.1f} MB".format(total_bytes / (1024 * 1024)))


if __name__ == "__main__":
    main()
