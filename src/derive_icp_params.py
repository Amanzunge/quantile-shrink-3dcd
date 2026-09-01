"""
Derive the icp_euclidean distance->score parameters (D0, S0, ICP max-correspondence)
for a dataset, the same way Task 4 fixed them for LD. Re-run per dataset because the
LD values do NOT transfer (Section 14 / Task 4 note): denser or sparser clouds shift
the no-change C2C distance floor.

Reasoning (label-free, matches Task 4): the C2C baseline scores each t1 point by its
nearest-neighbour distance to the aligned t0 cloud. For an UNCHANGED t1 point that
distance is bounded by how sparse the t0 cloud is, i.e. the t0 within-cloud spacing.
So the natural threshold is one t0 point spacing, with a half-spacing sigmoid scale:
    D0  = median within-cloud nearest-neighbour spacing of t0  (one point spacing)
    S0  = D0 / 2                                                (half a spacing)
    corr= 2 * D0                                                (ICP max correspondence)
tau = 0.5 then reproduces the classic "threshold the C2C distance at one spacing".

We report t1 spacing too, purely for context: on MS t1 is much denser than t0 (the
multi-sensor asymmetry), which is exactly why D0 must come from t0, not t1.

Spacing is translation invariant and scipy cKDTree works in float64, so no centering
is needed here (predict_icp.py centers only for ICP/open3d numerical stability).

  python src/derive_icp_params.py --config configs/urb3dcd_v2_ms.yaml
"""

import os
import sys
import json
import argparse
import numpy as np
import yaml
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inspect_dataset import read_cloud


def median_nn_spacing(xyz, max_queries, rng):
    """Median within-cloud nearest-neighbour distance. The KD-tree is built on the FULL
    cloud (so neighbours are exact); only the query set is subsampled for speed. k=2 takes
    the nearest OTHER point (k=1 would be the point itself at distance 0)."""
    tree = cKDTree(xyz)                                   # full cloud -> exact neighbours
    if len(xyz) > max_queries:
        q = xyz[rng.choice(len(xyz), size=max_queries, replace=False)]   # subsample queries only
    else:
        q = xyz
    dist, _ = tree.query(q, k=2)                          # [Q, 2]; col 0 is self (dist 0)
    nn = dist[:, 1]                                       # nearest distinct neighbour distance
    return float(np.median(nn))


def main():
    parser = argparse.ArgumentParser(description="Derive icp_euclidean D0/S0/corr per dataset.")
    parser.add_argument("--config", required=True, help="dataset YAML in configs/")
    parser.add_argument("--splits", nargs="*", default=["val"],
                        help="which splits to measure spacing on (default: val, the fit set)")
    parser.add_argument("--max-queries", type=int, default=200000,
                        help="cap on query points per cloud for the median (tree is always full)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    raw_root = cfg["raw_root"]
    ply_element = cfg.get("ply_element", "params")
    label_field = cfg.get("label_field", "label_ch")
    splits = json.load(open(cfg["splits_file"]))
    rng = np.random.default_rng(42)                       # fixed seed -> reproducible spacing estimate

    print("dataset:", cfg["dataset_name"], "| measuring spacing on splits:", args.splits)
    print("%-10s %-6s %12s %12s" % ("scene", "split", "t0_spacing", "t1_spacing"))
    t0_spacings = []                                      # pooled across scenes for the recommendation
    for split in args.splits:
        for scene_rel in splits.get(split, []):
            scene_id = scene_rel.split("/")[-1]
            scene_path = os.path.join(raw_root, *scene_rel.split("/"))
            xyz0, _ = read_cloud(os.path.join(scene_path, "pointCloud0.ply"), ply_element, label_field)
            xyz1, _ = read_cloud(os.path.join(scene_path, "pointCloud1.ply"), ply_element, label_field)
            s0_sp = median_nn_spacing(xyz0, args.max_queries, rng)
            s1_sp = median_nn_spacing(xyz1, args.max_queries, rng)
            t0_spacings.append(s0_sp)
            print("%-10s %-6s %12.4f %12.4f" % (scene_id, split, s0_sp, s1_sp))

    # Recommendation comes from the t0 spacing (the reference-cloud sampling gap).
    d0 = float(np.median(t0_spacings))
    s0 = d0 / 2.0
    corr = 2.0 * d0
    print("")
    print("median t0 spacing across measured scenes: %.4f m" % d0)
    print("RECOMMENDED icp_euclidean flags (Task 4 convention D0=one spacing, S0=half, corr=2x):")
    print("  --d0 %.3f --s0 %.3f --icp-corr %.3f" % (d0, s0, corr))


if __name__ == "__main__":
    main()
