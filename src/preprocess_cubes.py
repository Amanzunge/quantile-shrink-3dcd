"""
Task 2 cube-slicing preprocessor for Urb3DCD-V2 (and any later dataset that
shares the same PLY format). For each scene it tiles the shared XY bounding box
into cubes, drops cubes that do not hold enough points in BOTH clouds, and writes
ONE cache file per scene under data/cache/<dataset>/<scene_id>.npz.

The cache stores the cube STRUCTURE only: per-cube id, grid cell, spatial bbox,
the point INDICES into each raw cloud, and the per-cube label distribution.
It deliberately does NOT do FPS and does NOT copy XYZ. FPS happens later in the
dataloader (Task 3). Because the cache holds raw indices, one cache serves every
FPS target in the Section 5.1 sweep without re-slicing, and the "never upsample"
policy is decided where FPS actually lives.

The cube grid here is byte-for-byte the same grid as the Task 1 inventory
(src/inspect_dataset.py count_cubes), so the kept-cube totals reconcile exactly.

Run:
    python src/preprocess_cubes.py --config configs/urb3dcd_v2_ld.yaml
    python src/preprocess_cubes.py --config configs/urb3dcd_v2_ms.yaml
"""

import os
import sys
import json
import argparse
import numpy as np

# Make the sibling Task 1 reader importable no matter what the working dir is,
# so PLY parsing here is identical to the inventory tool (same element/label).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inspect_dataset import read_cloud, cube_keys


def load_split_scenes(splits_file):
    """
    Read the frozen splits.json and return a list of (split, scene_rel, scene_id).
    scene_rel is the path under raw_root ("Train/LyonN"); scene_id is the leaf.
    """
    with open(splits_file) as f:
        splits = json.load(f)
    scenes = []  # one (split, scene_rel, scene_id) tuple per scene, in split order
    for split in ["train", "val", "test"]:
        for scene_rel in splits.get(split, []):
            # The leaf folder name is unique within a dataset, so it is the id.
            scene_id = scene_rel.split("/")[-1]
            scenes.append((split, scene_rel, scene_id))
    return scenes


def group_indices_by_key(cell_key, kept_keys):
    """
    Given the per-point cube key of one cloud and the sorted list of kept cube
    keys, return a list of index arrays: groups[k] holds the ORIGINAL point
    indices of that cloud falling in kept cube k.

    Sorting the points by key once makes every cube a contiguous block, so each
    cube is found with two binary searches instead of a full scan per cube.
    """
    order = np.argsort(cell_key, kind="stable")   # stable: keep original order inside a cube
    sorted_key = cell_key[order]
    # First and one-past-last position of each kept key inside the sorted array.
    starts = np.searchsorted(sorted_key, kept_keys, side="left")
    ends = np.searchsorted(sorted_key, kept_keys, side="right")
    groups = []  # groups[k] = original indices of cloud points in kept cube k
    for start, end in zip(starts, ends):
        groups.append(order[start:end])
    return groups


def assign_cubes(xyz0, xyz1, cube_x, cube_y, min_points, cube_z=None):
    """
    Tile the shared bounding box into cubes and keep those with at least
    min_points in BOTH clouds. cube_z=None gives full-Z columns (outdoor, the
    original behaviour); a numeric cube_z also tiles the vertical axis (IndoorCD
    rooms). Returns a dict of arrays describing the kept cubes; cube ordering is
    by (ix, iy, iz) for a stable cube_id. Uses the same cube_keys helper as the
    Task 1 inventory, so kept-cube totals reconcile exactly.
    """
    # Shared grid origin: minimum XY over BOTH clouds (identical to Task 1).
    both_xy = np.concatenate([xyz0[:, :2], xyz1[:, :2]], axis=0)
    min_xy = both_xy.min(axis=0)

    # Per-point cube keys on the shared grid (XY, plus Z when cube_z is set).
    key0, key1, stride_y, stride_z, min_z = cube_keys(
        xyz0, xyz1, cube_x, cube_y, min_xy, cube_z)

    # Count points per cube in each cloud.
    uniq0, cnt0 = np.unique(key0, return_counts=True)
    uniq1, cnt1 = np.unique(key1, return_counts=True)
    count0_map = dict(zip(uniq0.tolist(), cnt0.tolist()))
    count1_map = dict(zip(uniq1.tolist(), cnt1.tolist()))

    # A kept cube must clear min_points in both clouds, so it lives in both maps.
    # Sorting the keys gives the canonical, reproducible cube_id ordering.
    kept_keys = []
    for key in sorted(set(count0_map) & set(count1_map)):
        if count0_map[key] >= min_points and count1_map[key] >= min_points:
            kept_keys.append(key)
    kept_keys = np.array(kept_keys, dtype=np.int64)
    num_cubes = len(kept_keys)

    # Recover the grid cell (ix, iy, iz) of each kept cube from its packed key
    # (inverse of cube_keys: key = ((ix*stride_y)+iy)*stride_z + iz).
    grid_iz = (kept_keys % stride_z).astype(np.int32)
    remainder = kept_keys // stride_z
    grid_iy = (remainder % stride_y).astype(np.int32)
    grid_ix = (remainder // stride_y).astype(np.int32)

    # Group the point indices of each cloud into the kept cubes.
    groups0 = group_indices_by_key(key0, kept_keys)
    groups1 = group_indices_by_key(key1, kept_keys)

    # Per-cube spatial bbox: XY from the grid tile; Z is the full scene range for
    # full-Z columns, or the cube_z-tall tile when the grid is tiled in Z.
    bbox_min = np.zeros((num_cubes, 3), dtype=np.float32)
    bbox_max = np.zeros((num_cubes, 3), dtype=np.float32)
    bbox_min[:, 0] = min_xy[0] + grid_ix * cube_x
    bbox_min[:, 1] = min_xy[1] + grid_iy * cube_y
    bbox_max[:, 0] = min_xy[0] + (grid_ix + 1) * cube_x
    bbox_max[:, 1] = min_xy[1] + (grid_iy + 1) * cube_y
    if cube_z is None:
        # Full-scene Z range; cubes span the whole Z (no vertical tiling).
        z_min = float(min(xyz0[:, 2].min(), xyz1[:, 2].min()))
        z_max = float(max(xyz0[:, 2].max(), xyz1[:, 2].max()))
        bbox_min[:, 2] = z_min
        bbox_max[:, 2] = z_max
    else:
        # Each cube occupies one cube_z-tall tile measured from the shared min Z.
        bbox_min[:, 2] = min_z + grid_iz * float(cube_z)
        bbox_max[:, 2] = min_z + (grid_iz + 1) * float(cube_z)

    # Per-cube point counts in each cloud (what Task 3 needs for the FPS policy).
    count_t0 = np.array([len(g) for g in groups0], dtype=np.int32)
    count_t1 = np.array([len(g) for g in groups1], dtype=np.int32)

    return {
        "min_xy": min_xy,
        "kept_keys": kept_keys,
        "grid_ix": grid_ix,
        "grid_iy": grid_iy,
        "grid_iz": grid_iz,
        "bbox_min": bbox_min,
        "bbox_max": bbox_max,
        "groups0": groups0,
        "groups1": groups1,
        "count_t0": count_t0,
        "count_t1": count_t1,
    }


def build_label_distribution(labels1, groups1, num_classes):
    """
    Per-cube label stats on the t1 cloud: a [K, num_classes] source histogram and
    the binary unchanged/changed split (class 0 vs classes 1..num_classes-1).
    Returns (class_hist int32, n_unchanged int32, n_changed int32).
    """
    num_cubes = len(groups1)
    class_hist = np.zeros((num_cubes, num_classes), dtype=np.int32)
    n_unchanged = np.zeros(num_cubes, dtype=np.int32)
    n_changed = np.zeros(num_cubes, dtype=np.int32)
    for cube in range(num_cubes):
        # Source labels of this cube's t1 points, clipped into the valid range.
        cube_labels = labels1[groups1[cube]].clip(0, num_classes - 1)
        hist = np.bincount(cube_labels, minlength=num_classes)
        class_hist[cube] = hist[:num_classes]
        n_unchanged[cube] = hist[0]            # class 0 is unchanged
        n_changed[cube] = hist[1:].sum()       # any of classes 1..6 is change
    return class_hist, n_unchanged, n_changed


def pack_csr(groups):
    """
    Pack a list of variable-length index arrays into one concatenated index array
    plus a CSR-style offset array of length K+1, so cube k's indices are
    indices[offset[k]:offset[k+1]]. Empty input yields empty/zero arrays.
    """
    if len(groups) == 0:
        return np.empty(0, dtype=np.int64), np.zeros(1, dtype=np.int64)
    indices = np.concatenate(groups).astype(np.int64)
    # Offsets are the cumulative sum of the per-cube lengths, prefixed with 0.
    lengths = np.array([len(g) for g in groups], dtype=np.int64)
    offset = np.zeros(len(groups) + 1, dtype=np.int64)
    offset[1:] = np.cumsum(lengths)
    return indices, offset


def process_scene(raw_root, scene_rel, scene_id, split, dataset_name,
                  cube_x, cube_y, min_points, ply_element, label_field,
                  num_classes, cache_root, cube_z=None):
    """
    Slice one scene into cubes and write its cache file. Returns a manifest row
    (a small dict) summarising the scene for the dataset-level manifest. cube_z
    is the vertical tile height (None = full-Z columns; a number tiles Z too).
    """
    # Build the on-disk scene path from the relative split/leaf entry.
    scene_path = os.path.join(raw_root, *scene_rel.split("/"))
    pc0_file = os.path.join(scene_path, "pointCloud0.ply")
    pc1_file = os.path.join(scene_path, "pointCloud1.ply")

    # Read both clouds with the exact same reader as Task 1.
    xyz0, _ = read_cloud(pc0_file, ply_element, label_field)
    xyz1, labels1 = read_cloud(pc1_file, ply_element, label_field)
    # The t1 cloud must carry change labels; fall back to zeros if missing.
    if labels1 is None:
        labels1 = np.zeros(len(xyz1), dtype=np.int64)

    # Tile into cubes and keep those with enough points in both clouds.
    cubes = assign_cubes(xyz0, xyz1, cube_x, cube_y, min_points, cube_z)
    num_cubes = len(cubes["kept_keys"])

    # Per-cube label distribution on the t1 cloud.
    class_hist, n_unchanged, n_changed = build_label_distribution(
        labels1, cubes["groups1"], num_classes)

    # Pack the per-cloud point indices into CSR (concatenated + offsets).
    idx_t0, offset_t0 = pack_csr(cubes["groups0"])
    idx_t1, offset_t1 = pack_csr(cubes["groups1"])

    # cube_id is just 0..K-1 in the canonical (ix, iy) order.
    cube_id = np.arange(num_cubes, dtype=np.int32)

    # Full-scene AABB over both clouds (origin for the grid and for normalisation).
    both_xyz = np.concatenate([xyz0, xyz1], axis=0)
    bbox_min_scene = both_xyz.min(axis=0).astype(np.float32)
    bbox_max_scene = both_xyz.max(axis=0).astype(np.float32)

    # Write the per-scene cache. Compressed because the index arrays are large.
    os.makedirs(cache_root, exist_ok=True)
    out_file = os.path.join(cache_root, scene_id + ".npz")
    np.savez_compressed(
        out_file,
        # scene-level metadata
        dataset=np.array(dataset_name),
        scene_id=np.array(scene_id),
        split=np.array(split),
        cube_size_xy=np.array([cube_x, cube_y], dtype=np.float32),
        # cube_size_z: -1.0 marks full-Z columns; a positive value is the tile height.
        cube_size_z=np.array(-1.0 if cube_z is None else float(cube_z), dtype=np.float32),
        min_points=np.array(min_points, dtype=np.int32),
        num_source_classes=np.array(num_classes, dtype=np.int32),
        n_points_t0=np.array(len(xyz0), dtype=np.int64),
        n_points_t1=np.array(len(xyz1), dtype=np.int64),
        scene_bbox_min=bbox_min_scene,
        scene_bbox_max=bbox_max_scene,
        grid_origin_xy=cubes["min_xy"].astype(np.float32),
        # per-cube structure (length K)
        cube_id=cube_id,
        cube_grid_ix=cubes["grid_ix"],
        cube_grid_iy=cubes["grid_iy"],
        cube_grid_iz=cubes["grid_iz"],
        cube_bbox_min=cubes["bbox_min"],
        cube_bbox_max=cubes["bbox_max"],
        count_t0=cubes["count_t0"],
        count_t1=cubes["count_t1"],
        n_unchanged=n_unchanged,
        n_changed=n_changed,
        class_hist=class_hist,
        # per-cloud point indices in CSR layout
        idx_t0=idx_t0,
        offset_t0=offset_t0,
        idx_t1=idx_t1,
        offset_t1=offset_t1,
    )

    # Manifest row: counts only, used for the dataset-level reconciliation.
    return {
        "scene_id": scene_id,
        "split": split,
        "scene_rel": scene_rel,
        "points_t0": int(len(xyz0)),
        "points_t1": int(len(xyz1)),
        "cubes_kept": int(num_cubes),
        "points_in_cubes_t0": int(idx_t0.shape[0]),
        "points_in_cubes_t1": int(idx_t1.shape[0]),
        "changed_points_t1": int(n_changed.sum()),
        "cache_file": os.path.relpath(out_file).replace("\\", "/"),
    }


def main():
    parser = argparse.ArgumentParser(description="Urb3DCD-V2 cube slicer (Task 2).")
    parser.add_argument("--config", required=True, help="dataset YAML in configs/")
    parser.add_argument("--cache-root", default=None,
                        help="override the cache output dir (used by ablations)")
    parser.add_argument("--cube-size", type=float, default=None,
                        help="override cube XY extent in metres (cube-size sweep)")
    args = parser.parse_args()

    # Read the dataset config (paths, cube size, FPS target, min points, format).
    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    dataset_name = cfg["dataset_name"]
    raw_root = cfg["raw_root"]
    cache_root = args.cache_root if args.cache_root is not None else cfg["cache_root"]
    splits_file = cfg["splits_file"]
    cube_x = float(cfg["cube_size_x"])
    cube_y = float(cfg["cube_size_y"])
    min_points = int(cfg["min_points"])
    # cube_size_z "full" (or absent) = full-Z outdoor columns; a number tiles Z
    # too (IndoorCD room-scale cubes are bounded vertically as well as in XY).
    cube_size_z = cfg.get("cube_size_z", "full")
    cube_z = None if cube_size_z == "full" else float(cube_size_z)
    ply_element = cfg.get("ply_element", "params")
    label_field = cfg.get("label_field", "label_ch")
    num_classes = int(cfg.get("num_source_classes", 7))

    # The cube-size sweep overrides the XY extent (applied to both axes).
    if args.cube_size is not None:
        cube_x = args.cube_size
        cube_y = args.cube_size

    scenes = load_split_scenes(splits_file)
    z_desc = "full Z" if cube_z is None else (str(cube_z) + " m Z tiles")
    print("Dataset:", dataset_name)
    print("Raw root:", raw_root)
    print("Cache root:", cache_root)
    print("Cube XY:", cube_x, "x", cube_y, "m,", z_desc, "; min_points:", min_points)
    print("Scenes to slice:", len(scenes))
    print("")

    # Per-scene processing with a one-line progress print each.
    header = "{:<6} {:<10} {:>10} {:>10} {:>8} {:>10} {:>10}".format(
        "split", "scene", "pts_t0", "pts_t1", "cubes", "kept_t0", "kept_t1")
    print(header)
    print("-" * len(header))

    manifest_rows = []
    for split, scene_rel, scene_id in scenes:
        row = process_scene(
            raw_root, scene_rel, scene_id, split, dataset_name,
            cube_x, cube_y, min_points, ply_element, label_field,
            num_classes, cache_root, cube_z=cube_z)
        manifest_rows.append(row)
        print("{:<6} {:<10} {:>10d} {:>10d} {:>8d} {:>10d} {:>10d}".format(
            split, scene_id[:10], row["points_t0"], row["points_t1"],
            row["cubes_kept"], row["points_in_cubes_t0"], row["points_in_cubes_t1"]))

    # Dataset-level totals, broken down by split for the Task 1 reconciliation.
    totals_by_split = {}
    for split in ["train", "val", "test"]:
        group = [r for r in manifest_rows if r["split"] == split]
        totals_by_split[split] = {
            "scenes": len(group),
            "cubes_kept": sum(r["cubes_kept"] for r in group),
        }
    total_cubes = sum(r["cubes_kept"] for r in manifest_rows)

    # Write the dataset manifest next to the per-scene caches.
    manifest = {
        "dataset": dataset_name,
        "cube_size_xy": [cube_x, cube_y],
        "cube_size_z": "full" if cube_z is None else cube_z,
        "min_points": min_points,
        "num_source_classes": num_classes,
        "scenes": manifest_rows,
        "totals_by_split": totals_by_split,
        "total_cubes_kept": total_cubes,
    }
    os.makedirs(cache_root, exist_ok=True)
    manifest_file = os.path.join(cache_root, "manifest.json")
    with open(manifest_file, "w") as f:
        json.dump(manifest, f, indent=2)

    print("")
    print("Kept cubes by split:",
          "train", totals_by_split["train"]["cubes_kept"],
          "val", totals_by_split["val"]["cubes_kept"],
          "test", totals_by_split["test"]["cubes_kept"])
    print("Total kept cubes:", total_cubes)
    print("Manifest written to:", manifest_file)


if __name__ == "__main__":
    main()
