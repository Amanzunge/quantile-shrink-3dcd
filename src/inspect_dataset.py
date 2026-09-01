"""
Task 1 inventory tool for Urb3DCD-V2 (LD and MS sub-datasets).

What it does, in order:
  1. Find every scene. A scene is any folder that holds both pointCloud0.ply
     (date t0) and pointCloud1.ply (date t1).
  2. Read XYZ and the change label (label_ch) from each cloud.
  3. Report per-scene point counts, bounding box, and binary change ratio.
  4. Project how many 50x50 metre cubes (full Z) each scene yields, and how many
     survive the "at least 256 points in BOTH clouds" drop rule from Section 6.
  5. Print per-split and per-dataset aggregates and write a per-scene CSV.

This is a read-only inventory tool. It never modifies the raw data.

First-look structure discovery on a freshly unzipped folder:
    python src/inspect_dataset.py --root data/raw/IEEE_Dataset_V2_Lid05_MS

Once the data is sorted into data/raw/<dataset>/ and the config points at it:
    python src/inspect_dataset.py --config configs/urb3dcd_v2_ld.yaml --csv results/urb3dcd_v2_ld_inventory.csv
"""

import os
import csv
import argparse
import numpy as np

# plyfile reads the PLY files the dataset ships (ascii or binary).
from plyfile import PlyData


def find_scenes(root):
    """Return a sorted list of folders that contain both point cloud files."""
    scenes = []  # each entry is a folder path holding the pointCloud pair
    # Walk the whole tree once; a scene is any directory with the pair in it.
    for current_dir, sub_dirs, files in os.walk(root):
        if "pointCloud0.ply" in files and "pointCloud1.ply" in files:
            scenes.append(current_dir)
    scenes.sort()
    return scenes


def guess_split(scene_path):
    """Guess train/val/test from the folder path; 'unknown' if not obvious."""
    lowered = scene_path.lower()
    # Test before train/val so a path like ".../test/..." is not misread.
    if os.sep + "test" in lowered or "/test" in lowered:
        return "test"
    if os.sep + "val" in lowered or "/val" in lowered:
        return "val"
    if os.sep + "train" in lowered or "/train" in lowered:
        return "train"
    return "unknown"


def read_cloud(ply_path, ply_element, label_field):
    """Read one PLY. Return (xyz float32 [N,3], labels int64 [N] or None)."""
    ply = PlyData.read(ply_path)
    element_names = [e.name for e in ply.elements]
    # Prefer the configured element name; otherwise take the first element
    # that actually carries x, y, z fields (robust to naming differences).
    if ply_element in element_names:
        element = ply[ply_element]
    else:
        element = None
        for e in ply.elements:
            field_names = e.data.dtype.names
            if field_names is not None and "x" in field_names and "y" in field_names and "z" in field_names:
                element = e
                break
        if element is None:
            raise ValueError("No PLY element with x,y,z found in " + ply_path)
    data = element.data
    # Stack the three coordinate fields into one [N, 3] float32 array.
    xyz = np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float32)
    # Read the change label only if the field exists (t0 cloud may lack it).
    labels = None
    if data.dtype.names is not None and label_field in data.dtype.names:
        labels = np.asarray(data[label_field]).astype(np.int64)
    return xyz, labels


def cube_keys(xyz0, xyz1, cube_x, cube_y, min_xy, cube_z=None):
    """
    Map every point of both clouds to an integer cube key on the shared grid.
    XY is always tiled into cube_x by cube_y columns. cube_z is the vertical tile
    height: None means full-Z columns (one Z tile, outdoor LD/MS/HKCD); a number
    means the grid is also tiled in Z (indoor IndoorCD rooms). The key packs
    (ix, iy, iz) so sorting keys yields a canonical (ix, iy, iz) cube ordering.

    Returns (key0, key1, stride_y, stride_z, min_z). With cube_z=None the key
    reduces to ix*stride_y + iy, byte-for-byte the old XY-only key, so existing
    outdoor caches are unchanged.
    """
    # Integer (ix, iy) cube index per point by flooring onto the XY grid.
    cell0_xy = np.floor((xyz0[:, :2] - min_xy) / np.array([cube_x, cube_y], dtype=np.float64)).astype(np.int64)
    cell1_xy = np.floor((xyz1[:, :2] - min_xy) / np.array([cube_x, cube_y], dtype=np.float64)).astype(np.int64)
    stride_y = int(max(cell0_xy[:, 1].max(), cell1_xy[:, 1].max())) + 1
    if cube_z is None:
        # Full-Z columns: a single Z tile (iz = 0), so stride_z = 1 collapses the
        # key back to the original XY-only key.
        iz0 = np.zeros(len(xyz0), dtype=np.int64)
        iz1 = np.zeros(len(xyz1), dtype=np.int64)
        stride_z = 1
        min_z = float(min(xyz0[:, 2].min(), xyz1[:, 2].min()))
    else:
        # 3D tiling: floor Z onto cube_z tiles from the shared minimum Z.
        min_z = float(min(xyz0[:, 2].min(), xyz1[:, 2].min()))
        iz0 = np.floor((xyz0[:, 2] - min_z) / float(cube_z)).astype(np.int64)
        iz1 = np.floor((xyz1[:, 2] - min_z) / float(cube_z)).astype(np.int64)
        stride_z = int(max(iz0.max(), iz1.max())) + 1
    # Pack (ix, iy, iz) into one key: ((ix*stride_y)+iy)*stride_z + iz.
    key0 = (cell0_xy[:, 0] * stride_y + cell0_xy[:, 1]) * stride_z + iz0
    key1 = (cell1_xy[:, 0] * stride_y + cell1_xy[:, 1]) * stride_z + iz1
    return key0, key1, stride_y, stride_z, min_z


def count_cubes(xyz0, xyz1, cube_x, cube_y, min_points, cube_z=None):
    """
    Tile the shared bounding box into cubes (full-Z columns when cube_z is None,
    else cube_z-tall 3D cubes). Return (occupied_cubes, kept_cubes, kept_counts0,
    kept_counts1) where kept means both clouds have at least min_points in that
    cube (Section 6 drop rule), and the two arrays hold the per-cube point counts
    of the kept cubes for the t0 and t1 clouds.
    """
    # Use a bounding box over BOTH clouds so the two dates share one cube grid.
    all_xy = np.concatenate([xyz0[:, :2], xyz1[:, :2]], axis=0)
    min_xy = all_xy.min(axis=0)
    # Per-point cube keys on the shared grid (XY, plus Z when cube_z is set).
    key0, key1, _, _, _ = cube_keys(xyz0, xyz1, cube_x, cube_y, min_xy, cube_z)
    # Count how many points fall in each cube, separately for each cloud.
    keys0, counts0 = np.unique(key0, return_counts=True)
    keys1, counts1 = np.unique(key1, return_counts=True)
    count0_map = dict(zip(keys0.tolist(), counts0.tolist()))
    count1_map = dict(zip(keys1.tolist(), counts1.tolist()))
    # Occupied cubes = union of cube keys touched by either cloud.
    all_keys = set(count0_map.keys()) | set(count1_map.keys())
    kept_counts0 = []  # per-cube t0 point counts, kept cubes only
    kept_counts1 = []  # per-cube t1 point counts, kept cubes only
    for k in all_keys:
        c0 = count0_map.get(k, 0)
        c1 = count1_map.get(k, 0)
        # Keep only if BOTH clouds clear the minimum point count.
        if c0 >= min_points and c1 >= min_points:
            kept_counts0.append(c0)
            kept_counts1.append(c1)
    return (len(all_keys), len(kept_counts0),
            np.array(kept_counts0, dtype=np.int64),
            np.array(kept_counts1, dtype=np.int64))


def main():
    parser = argparse.ArgumentParser(description="Urb3DCD-V2 inventory (Task 1).")
    parser.add_argument("--config", default=None, help="dataset YAML in configs/")
    parser.add_argument("--root", default=None, help="folder to scan directly")
    parser.add_argument("--csv", default=None, help="optional per-scene CSV output")
    parser.add_argument("--num-classes", type=int, default=7, help="source label classes")
    parser.add_argument("--cube-size", type=float, default=None,
                        help="override cube XY extent in metres (Task 12 cube-size sweep inventory); "
                             "Z tiling (cube_size_z) stays from the config")
    args = parser.parse_args()

    # Defaults match the frozen Section 4 settings and the verified PLY format.
    root = args.root
    cube_x, cube_y, min_points = 50.0, 50.0, 256
    cube_z = None  # None = full-Z columns (outdoor); a number tiles Z (IndoorCD)
    ply_element, label_field = "params", "label_ch"
    num_classes = args.num_classes

    # A config file overrides the defaults and provides the raw_root path.
    if args.config is not None:
        import yaml
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        root = cfg["raw_root"]
        cube_x = float(cfg["cube_size_x"])
        cube_y = float(cfg["cube_size_y"])
        min_points = int(cfg["min_points"])
        # cube_size_z is "full" (or absent) for full-Z outdoor columns, else a
        # numeric vertical tile height in metres (IndoorCD tiles Z as well as XY).
        cube_size_z = cfg.get("cube_size_z", "full")
        cube_z = None if cube_size_z == "full" else float(cube_size_z)
        ply_element = cfg.get("ply_element", "params")
        label_field = cfg.get("label_field", "label_ch")
        num_classes = int(cfg.get("num_source_classes", num_classes))

    # Cube-size sweep override (Task 12): set both XY axes; matches preprocess_cubes --cube-size.
    if args.cube_size is not None:
        cube_x = args.cube_size
        cube_y = args.cube_size

    if root is None:
        parser.error("provide either --config or --root")
    if not os.path.isdir(root):
        parser.error("root folder does not exist: " + root)

    scenes = find_scenes(root)
    print("Scanning:", root)
    print("Scenes found (folders with the pointCloud pair):", len(scenes))
    if len(scenes) == 0:
        print("No scenes found. Check the path or the unzip location.")
        return

    print("")
    header = "{:<6} {:<28} {:>10} {:>10} {:>8} {:>8} {:>9} {:>7} {:>7} {:>7}".format(
        "split", "scene", "pts_t0", "pts_t1", "extX", "extY", "chg%", "cubes", "kept", "drop")
    print(header)
    print("-" * len(header))

    rows = []  # collected per-scene records for the CSV
    # Per-cube point counts of kept cubes, gathered per split for a density summary.
    density = {"train": {"t0": [], "t1": []}, "val": {"t0": [], "t1": []},
               "test": {"t0": [], "t1": []}, "unknown": {"t0": [], "t1": []}}
    for scene_path in scenes:
        split = guess_split(scene_path)
        # Short scene name: the folder plus its parent for readability.
        parent = os.path.basename(os.path.dirname(scene_path))
        leaf = os.path.basename(scene_path)
        scene_name = parent + "/" + leaf

        pc0_file = os.path.join(scene_path, "pointCloud0.ply")
        pc1_file = os.path.join(scene_path, "pointCloud1.ply")
        xyz0, _ = read_cloud(pc0_file, ply_element, label_field)
        xyz1, labels1 = read_cloud(pc1_file, ply_element, label_field)

        # Bounding box over both clouds (this is the cube tiling extent).
        all_xyz = np.concatenate([xyz0, xyz1], axis=0)
        bbox_min = all_xyz.min(axis=0)
        bbox_max = all_xyz.max(axis=0)
        extent = bbox_max - bbox_min

        # Binary change ratio and per-class histogram on the t1 labels.
        change_ratio = 0.0
        class_counts = [0] * num_classes
        if labels1 is not None and len(labels1) > 0:
            change_ratio = float((labels1 > 0).sum()) / float(len(labels1))
            hist = np.bincount(labels1.clip(0, num_classes - 1), minlength=num_classes)
            class_counts = hist[:num_classes].tolist()

        # Projected cube counts at the dataset cube size and drop rule.
        occupied, kept, kept_c0, kept_c1 = count_cubes(xyz0, xyz1, cube_x, cube_y, min_points, cube_z)
        dropped = occupied - kept
        # Median points per kept cube, per cloud (0 if the scene kept no cube).
        med_c0 = int(np.median(kept_c0)) if kept > 0 else 0
        med_c1 = int(np.median(kept_c1)) if kept > 0 else 0
        # Stash the kept-cube counts so the per-split density block can use them.
        if kept > 0:
            density[split]["t0"].append(kept_c0)
            density[split]["t1"].append(kept_c1)

        print("{:<6} {:<28} {:>10d} {:>10d} {:>8.1f} {:>8.1f} {:>8.2f}% {:>7d} {:>7d} {:>7d}".format(
            split, scene_name[:28], len(xyz0), len(xyz1),
            extent[0], extent[1], 100.0 * change_ratio, occupied, kept, dropped))

        record = {
            "split": split,
            "scene": scene_name,
            "path": scene_path,
            "pts_t0": len(xyz0),
            "pts_t1": len(xyz1),
            "xmin": float(bbox_min[0]), "ymin": float(bbox_min[1]), "zmin": float(bbox_min[2]),
            "xmax": float(bbox_max[0]), "ymax": float(bbox_max[1]), "zmax": float(bbox_max[2]),
            "ext_x": float(extent[0]), "ext_y": float(extent[1]), "ext_z": float(extent[2]),
            "change_ratio": change_ratio,
            "cubes_occupied": occupied,
            "cubes_kept": kept,
            "cubes_dropped": dropped,
            "med_pts_cube_t0": med_c0,
            "med_pts_cube_t1": med_c1,
        }
        for c in range(num_classes):
            record["class_" + str(c)] = class_counts[c]
        rows.append(record)

    # Per-split and overall aggregates.
    print("")
    print("Aggregates by split:")
    print("{:<8} {:>7} {:>14} {:>14} {:>9} {:>9} {:>9}".format(
        "split", "scenes", "pts_t0", "pts_t1", "cubes", "kept", "drop"))
    for split in ["train", "val", "test", "unknown"]:
        group = [r for r in rows if r["split"] == split]
        if len(group) == 0:
            continue
        print("{:<8} {:>7d} {:>14d} {:>14d} {:>9d} {:>9d} {:>9d}".format(
            split, len(group),
            sum(r["pts_t0"] for r in group),
            sum(r["pts_t1"] for r in group),
            sum(r["cubes_occupied"] for r in group),
            sum(r["cubes_kept"] for r in group),
            sum(r["cubes_dropped"] for r in group)))
    print("{:<8} {:>7d} {:>14d} {:>14d} {:>9d} {:>9d} {:>9d}".format(
        "TOTAL", len(rows),
        sum(r["pts_t0"] for r in rows),
        sum(r["pts_t1"] for r in rows),
        sum(r["cubes_occupied"] for r in rows),
        sum(r["cubes_kept"] for r in rows),
        sum(r["cubes_dropped"] for r in rows)))

    # Per-cube point density of kept cubes. A median below the dataset FPS target
    # means FPS would have to upsample that cloud (not allowed, see Section 5.1).
    print("")
    print("Per-cube point density of kept cubes (points per cube, t0=pointCloud0, t1=pointCloud1):")
    print("{:<8} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9}".format(
        "split", "t0_p10", "t0_med", "t0_p90", "t0_max", "t1_p10", "t1_med", "t1_p90", "t1_max"))
    split_order = ["train", "val", "test", "unknown"]
    for split in split_order + ["TOTAL"]:
        if split == "TOTAL":
            list0 = [a for s in split_order for a in density[s]["t0"]]
            list1 = [a for s in split_order for a in density[s]["t1"]]
        else:
            list0 = density[split]["t0"]
            list1 = density[split]["t1"]
        if len(list0) == 0:
            continue
        c0 = np.concatenate(list0)
        c1 = np.concatenate(list1)
        print("{:<8} {:>9d} {:>9d} {:>9d} {:>9d} {:>9d} {:>9d} {:>9d} {:>9d}".format(
            split,
            int(np.percentile(c0, 10)), int(np.median(c0)), int(np.percentile(c0, 90)), int(c0.max()),
            int(np.percentile(c1, 10)), int(np.median(c1)), int(np.percentile(c1, 90)), int(c1.max())))

    # Write the per-scene CSV if a path was given.
    if args.csv is not None:
        os.makedirs(os.path.dirname(args.csv), exist_ok=True)
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print("")
        print("Per-scene CSV written to:", args.csv)


if __name__ == "__main__":
    main()
