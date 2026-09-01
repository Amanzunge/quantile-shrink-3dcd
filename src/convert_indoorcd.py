"""
Convert IndoorCD (Task 11) into the PLY pair layout the rest of the pipeline
already reads, so preprocess_cubes / dataloader / train / predict / calibrate run
unchanged (the only other code change is 3D Z-tiling in the cube grid).

IndoorCD is unlike Urb3DCD/HKCD on disk:
  - clouds are .pcd (binary_compressed, fields x y z normal_x normal_y normal_z
    rgb) -> plyfile cannot read them, so we read with Open3D and keep XYZ only;
  - labels are 3D bounding BOXES in Label/<scene>-N.json (object "name" is "Add"
    or "Remove", 8 cuboid corner vertices), NOT per-point labels;
  - a scene has a reference scan <scene>-1.pcd plus modified scans <scene>-N.pcd,
    so each (scan1, scanN) pair is one bi-temporal sample id "<scene>-N".

Per-point label rule for PC_t2 (= scanN, the t1 cloud), matching the Section 8
binary-change-on-t1 contract: a t1 point is CHANGE (1) if it lies inside ANY
annotated box (Add OR Remove), else UNCHANGED (0). We treat every box uniformly
as a change region. Measured membership justifies this: Add boxes hold the new
object's t1 points (median ~470, always > 0); Remove boxes hold the revealed
surface / residual t1 points where the object used to be (median ~140). Using
Add-only would make the 197 remove-only pairs degenerate (zero positives), which
would bias the scope-limit calibration; the uniform rule avoids that.

Boxes are axis-aligned in the shared (pre-aligned) frame, so we use the corner
AABB (min/max over the 8 vertices); for any slightly oriented box this is a safe
superset.

Output: data/raw/indoorcd/<Folder>/<sample_id>/pointCloud0.ply (t0 = scan1, XYZ)
and pointCloud1.ply (t1 = scanN, XYZ + int32 'change'), element name 'vertex'.

Run (after freeze_indoorcd_split.py):
    python src/convert_indoorcd.py
"""

import os
import json
import argparse
import numpy as np
import open3d as o3d
from plyfile import PlyData, PlyElement

RAW_IN = "data/raw/IndoorCD"
DATA_DIR = os.path.join(RAW_IN, "Data")
LABEL_DIR = os.path.join(RAW_IN, "Label")
SPLITS_FILE = "data/splits/indoorcd/splits.json"
# NOTE: a distinct name (not "indoorcd") so it does not collide with the source
# "data/raw/IndoorCD" on case-insensitive filesystems (Windows).
RAW_OUT = "data/raw/indoorcd_ply"


def read_pcd_xyz(path):
    """Read a (binary_compressed) PCD via Open3D, return XYZ float32 [N,3]."""
    pc = o3d.io.read_point_cloud(path)
    return np.asarray(pc.points, dtype=np.float32)


def box_aabb(vertices):
    """AABB (min_xyz, max_xyz) from a box's 8 corner vertices."""
    v = np.asarray(vertices, dtype=np.float64)
    return v.min(axis=0), v.max(axis=0)


def labels_from_boxes(xyz1, objects):
    """
    Binary change label per t1 point: 1 if inside ANY annotated box, else 0.
    objects is the JSON "objects" list (each has "name" and "vertices").
    """
    changed = np.zeros(len(xyz1), dtype=np.int32)  # 0 = unchanged everywhere first
    for obj in objects:
        box_min, box_max = box_aabb(obj["vertices"])
        # A point is inside the AABB if it is within [min, max] on all three axes.
        inside = np.all((xyz1 >= box_min) & (xyz1 <= box_max), axis=1)
        changed[inside] = 1  # mark every point that falls in this change region
    return changed


def write_ply_xyz(path, xyz, change=None):
    """
    Write a binary PLY with element 'vertex'. Fields are x,y,z float32, plus an
    int32 'change' field when labels are given (the t1 cloud).
    """
    if change is None:
        verts = np.empty(len(xyz), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    else:
        verts = np.empty(len(xyz),
                         dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("change", "i4")])
        verts["change"] = change
    verts["x"] = xyz[:, 0]
    verts["y"] = xyz[:, 1]
    verts["z"] = xyz[:, 2]
    element = PlyElement.describe(verts, "vertex")
    PlyData([element], text=False).write(path)  # binary, native (little-endian) order


def main():
    parser = argparse.ArgumentParser(description="Convert IndoorCD to PLY pairs (Task 11).")
    parser.add_argument("--limit", type=int, default=None,
                        help="convert only the first N pairs (smoke test)")
    args = parser.parse_args()

    with open(SPLITS_FILE) as f:
        splits = json.load(f)

    # One flat list of (split_folder, sample_id) to convert.
    to_convert = []
    for split in ["train", "val", "test"]:
        for entry in splits[split]:
            folder, sample_id = entry.split("/")
            to_convert.append((folder, sample_id))
    if args.limit is not None:
        to_convert = to_convert[:args.limit]

    print("pairs to convert:", len(to_convert))
    print("output root:", RAW_OUT)
    print("")

    change_rates = []   # per-pair fraction of changed t1 points (sanity summary)
    n_done = 0
    for folder, sample_id in to_convert:
        scene = sample_id.split("-")[0]              # "001" from "001-2"
        ref_pcd = os.path.join(DATA_DIR, scene, scene + "-1.pcd")   # t0
        mod_pcd = os.path.join(DATA_DIR, scene, sample_id + ".pcd")  # t1
        label_json = os.path.join(LABEL_DIR, sample_id + ".json")

        # Skip (loudly) if any input is missing rather than guessing.
        if not (os.path.exists(ref_pcd) and os.path.exists(mod_pcd) and os.path.exists(label_json)):
            print("SKIP missing input:", sample_id)
            continue

        xyz0 = read_pcd_xyz(ref_pcd)
        xyz1 = read_pcd_xyz(mod_pcd)
        with open(label_json) as f:
            meta = json.load(f)
        change = labels_from_boxes(xyz1, meta.get("objects", []))
        change_rates.append(float(change.mean()))

        # Write the pair into data/raw/indoorcd/<Folder>/<sample_id>/.
        out_dir = os.path.join(RAW_OUT, folder, sample_id)
        os.makedirs(out_dir, exist_ok=True)
        write_ply_xyz(os.path.join(out_dir, "pointCloud0.ply"), xyz0)               # t0, no label
        write_ply_xyz(os.path.join(out_dir, "pointCloud1.ply"), xyz1, change=change)  # t1, labelled

        n_done += 1
        if n_done % 100 == 0:
            print("  converted", n_done, "/", len(to_convert))

    rates = np.asarray(change_rates, dtype=np.float64)
    print("")
    print("converted pairs:", n_done)
    if len(rates) > 0:
        print("t1 change rate: min {:.3f}  p10 {:.3f}  median {:.3f}  p90 {:.3f}  max {:.3f}".format(
            rates.min(), np.percentile(rates, 10), np.median(rates),
            np.percentile(rates, 90), rates.max()))
        print("pairs with zero changed points:", int((rates == 0).sum()))


if __name__ == "__main__":
    main()
