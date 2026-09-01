"""
Task 4: ICP + Euclidean cloud-to-cloud change-detection baseline for Urb3DCD-V2 LD.

This is the geometric baseline `icp_euclidean` from Section 2. It has NO training
and NO FPS step: it runs at full cube resolution on the original metric XYZ. For
each kept cube it rigidly aligns the t0 cloud onto the t1 cloud with point-to-point
ICP, then scores every t1 point by its nearest-neighbour distance to the aligned t0
cloud. A large distance means the t1 point has no t0 counterpart, i.e. change.

The cube-level scope matches the cube pipeline (Section 6) and Section 5.2 ("cube
extent changes registration scope"). t1 stays the reference cloud; its original
unnormalised coordinates are what we save, per the Section 8 output contract.

Distance -> (logits, scores) mapping (FROZEN, label-free, monotonic in distance):
    logits[:, 0] = 0
    logits[:, 1] = (d - D0) / S0
    scores       = sigmoid((d - D0) / S0) = softmax(logits)[:, 1]
so logits and scores are self-consistent (Task 7 platt/temperature act on logits,
the others on scores). D0 is the C2C distance threshold (a t1 point with no t0
neighbour within D0 metres is change) and S0 is the sigmoid scale. For LD both are
fixed from the dataset point spacing (~1.4 m median within-cloud NN): D0 = one point
spacing, S0 = half a point spacing. tau = 0.5 then reproduces the classic
"threshold the C2C distance at D0" baseline. Override --d0/--s0 per dataset later.

Numerical note: ICP is run in a per-cube CENTERED frame (both clouds shifted by their
common centroid) because the dataset uses a projected CRS with coordinates in the
millions; without centering a tiny rotation about the far world origin yields a
huge translation and unstable fits. Distance is translation-invariant, so the C2C
scores are unaffected and the saved coords stay original.

Output: predictions/urb3dcd_v2_ld/icp_euclidean/<scene_id>.npz (Section 8 schema)
for every val and test scene, plus predict_icp.log. No checkpoint, no train log.

Run:
    python src/predict_icp.py --config configs/urb3dcd_v2_ld.yaml
"""

import os
import sys
import json
import argparse
import logging
import numpy as np
import yaml
import open3d as o3d
from scipy.spatial import cKDTree
from scipy.special import expit

# Same PLY reader as Task 1/2/3 so coordinates and labels match the cache indices.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inspect_dataset import read_cloud

# Frozen LD defaults (metres). See the module docstring for the justification.
DEFAULT_D0 = 1.4        # C2C distance threshold ~= one LD point spacing
DEFAULT_S0 = 0.7        # sigmoid scale ~= half a point spacing
DEFAULT_CORR = 3.0      # ICP max correspondence distance ~= two point spacings
DEFAULT_ITER = 50       # ICP iteration cap


def setup_logger(log_path):
    """Log to both the predict_icp.log file and the console."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = logging.getLogger("predict_icp")
    logger.setLevel(logging.INFO)
    logger.handlers = []                                 # avoid duplicate handlers on re-run
    fmt = logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S")
    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)
    return logger


def load_val_test_scenes(splits_file):
    """Return [(split, scene_rel, scene_id), ...] for the val and test splits only."""
    with open(splits_file) as f:
        splits = json.load(f)
    scenes = []
    # ICP fits nothing on train, so only val (calibration fit) and test are needed.
    for split in ["val", "test"]:
        for scene_rel in splits.get(split, []):
            scene_id = scene_rel.split("/")[-1]          # leaf folder is the scene id
            scenes.append((split, scene_rel, scene_id))
    return scenes


def icp_align_t0(c0_centered, c1_centered, max_corr, max_iter):
    """
    Point-to-point ICP aligning source t0 onto target t1, both already centered.
    Returns (aligned t0 points [M0,3] float64, fitness). Falls back to the input
    points if the transform comes back non-finite (degenerate / no correspondences).
    """
    pcd0 = o3d.geometry.PointCloud()
    pcd0.points = o3d.utility.Vector3dVector(c0_centered)
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(c1_centered)
    result = o3d.pipelines.registration.registration_icp(
        pcd0, pcd1, max_corr, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter))
    transform = np.asarray(result.transformation)
    # Guard against a degenerate fit: keep the unaligned (identity) points instead.
    if not np.all(np.isfinite(transform)):
        return c0_centered, 0.0
    aligned = (c0_centered @ transform[:3, :3].T) + transform[:3, 3]
    return aligned, float(result.fitness)


def predict_scene(scene_rel, scene_id, raw_root, cache_root,
                  d0, s0, max_corr, max_iter, ply_element, label_field, dataset_name):
    """
    Run the ICP+Euclidean baseline on one scene. Returns (arrays dict, stats dict).
    Cubes are processed in cache order 0..K-1, and within a cube the t1 points keep
    their cache index order, so the concatenated output equals xyz1[idx_t1] exactly
    (this is what makes N reconcile with sum(count_t1)).
    """
    # Load the Task-2 cube cache (CSR point indices + per-cube boxes/counts).
    cache = np.load(os.path.join(cache_root, scene_id + ".npz"), allow_pickle=False)
    idx_t0, offset_t0 = cache["idx_t0"], cache["offset_t0"]
    idx_t1, offset_t1 = cache["idx_t1"], cache["offset_t1"]
    count_t1 = cache["count_t1"]
    num_cubes = len(cache["cube_id"])

    # Read both raw clouds once; the cache indices address these arrays directly.
    scene_path = os.path.join(raw_root, *scene_rel.split("/"))
    xyz0, _ = read_cloud(os.path.join(scene_path, "pointCloud0.ply"), ply_element, label_field)
    xyz1, labels1 = read_cloud(os.path.join(scene_path, "pointCloud1.ply"), ply_element, label_field)
    if labels1 is None:
        labels1 = np.zeros(len(xyz1), dtype=np.int64)
    binary_labels1 = (labels1 > 0).astype(np.int64)      # collapse 0..6 -> {0,1}

    # Per-cube outputs, concatenated at the end in cube order.
    out_scores, out_logits, out_labels, out_coords, out_cube_id = [], [], [], [], []
    fitnesses = []                                       # ICP fitness per cube, for the log

    for cube in range(num_cubes):
        # Original point indices of this cube in each cloud (cache CSR slice).
        sel0 = idx_t0[offset_t0[cube]:offset_t0[cube + 1]]
        sel1 = idx_t1[offset_t1[cube]:offset_t1[cube + 1]]
        cube_xyz0 = xyz0[sel0].astype(np.float64)        # float64 for ICP/KDTree precision
        cube_xyz1 = xyz1[sel1].astype(np.float64)

        # Center both clouds by their common centroid so ICP is numerically sound.
        offset = np.concatenate([cube_xyz0, cube_xyz1], axis=0).mean(axis=0)
        c0 = cube_xyz0 - offset
        c1 = cube_xyz1 - offset

        # Align t0 onto t1, then C2C nearest-neighbour distance for each t1 point.
        c0_aligned, fitness = icp_align_t0(c0, c1, max_corr, max_iter)
        fitnesses.append(fitness)
        tree = cKDTree(c0_aligned)
        dist, _ = tree.query(c1, k=1)                    # [M1] distance t1 -> aligned t0

        # Frozen distance -> logits -> scores map (label-free, monotonic in dist).
        logit_pos = ((dist - d0) / s0).astype(np.float32)   # logits[:,1]
        logits = np.zeros((len(dist), 2), dtype=np.float32)
        logits[:, 1] = logit_pos                            # logits[:,0] stays 0
        scores = expit(logit_pos).astype(np.float32)        # = softmax(logits)[:,1]

        out_scores.append(scores)
        out_logits.append(logits)
        out_labels.append(binary_labels1[sel1])             # GT in cache order
        out_coords.append(xyz1[sel1].astype(np.float32))    # ORIGINAL unnormalised t1 XYZ
        out_cube_id.append(np.full(len(dist), cube, dtype=np.int32))

    arrays = {
        "scores": np.concatenate(out_scores).astype(np.float32),
        "logits": np.concatenate(out_logits).astype(np.float32),
        "labels": np.concatenate(out_labels).astype(np.int64),
        "coords": np.concatenate(out_coords).astype(np.float32),
        "cube_id": np.concatenate(out_cube_id).astype(np.int32),
        "scene_id": scene_id,
        "dataset": dataset_name,                         # per-config (MS/HKCD), not hardcoded
    }

    # Sanity stats for the log: F1 at tau=0.5 (the fixed_05 operating point).
    pred = (arrays["scores"] > 0.5).astype(np.int64)
    gt = arrays["labels"]
    true_pos = int(((pred == 1) & (gt == 1)).sum())
    false_pos = int(((pred == 1) & (gt == 0)).sum())
    false_neg = int(((pred == 0) & (gt == 1)).sum())
    precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0.0
    recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    stats = {
        "num_points": len(arrays["scores"]),
        "sum_count_t1": int(count_t1.sum()),             # must equal num_points
        "num_cubes": num_cubes,
        "change_ratio": float((gt == 1).mean()),
        "icp_fitness_median": float(np.median(fitnesses)),
        "icp_fitness_min": float(np.min(fitnesses)),
        "f1_at_0p5": f1,
        "precision_at_0p5": precision,
        "recall_at_0p5": recall,
    }
    return arrays, stats


def verify_outputs(out_dir, scene_to_sum_count, dataset_name):
    """
    Section 8 schema-verification cell, plus the per-scene N == sum(count_t1) check.
    Raises AssertionError on any violation. Returns the number of files checked.
    """
    import glob
    files = sorted(glob.glob(os.path.join(out_dir, "*.npz")))
    assert len(files) > 0, "no .npz files written"
    for path in files:
        d = np.load(path, allow_pickle=True)
        for key in ["scores", "logits", "labels", "coords", "cube_id", "scene_id", "dataset"]:
            assert key in d.files, "missing key " + key + " in " + path
        assert d["scores"].dtype == np.float32 and d["scores"].ndim == 1
        assert d["logits"].dtype == np.float32 and d["logits"].shape[1] == 2
        assert d["labels"].dtype == np.int64
        assert d["coords"].dtype == np.float32 and d["coords"].shape[1] == 3
        assert d["cube_id"].dtype == np.int32
        n = len(d["scores"])
        assert n == len(d["labels"]) == len(d["coords"]) == len(d["cube_id"]) == len(d["logits"])
        # Extra Task-4 check: N must equal the cache's total kept t1 points.
        scene_id = str(d["scene_id"])
        assert n == scene_to_sum_count[scene_id], (
            "N mismatch for " + scene_id + ": npz " + str(n)
            + " vs cache sum(count_t1) " + str(scene_to_sum_count[scene_id]))
        assert str(d["dataset"]) == dataset_name
    return len(files)


def main():
    parser = argparse.ArgumentParser(description="ICP+Euclidean baseline (Task 4).")
    parser.add_argument("--config", required=True, help="dataset YAML in configs/")
    parser.add_argument("--out-root", default=None,
                        help="override output dir (default predictions/<dataset>/icp_euclidean)")
    parser.add_argument("--cache-root", default=None,
                        help="override the cube cache dir for the cube-size sweep (Section 5.2); ICP has no "
                             "FPS step so it appears only in the cube-size sweep, never the FPS sweep.")
    parser.add_argument("--d0", type=float, default=DEFAULT_D0, help="C2C distance threshold (m)")
    parser.add_argument("--s0", type=float, default=DEFAULT_S0, help="sigmoid scale (m)")
    parser.add_argument("--icp-corr", type=float, default=DEFAULT_CORR,
                        help="ICP max correspondence distance (m)")
    parser.add_argument("--icp-iter", type=int, default=DEFAULT_ITER, help="ICP iteration cap")
    args = parser.parse_args()

    # Read paths and PLY format from the dataset config.
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    dataset_name = cfg["dataset_name"]
    raw_root = cfg["raw_root"]
    # --cache-root override drives the cube-size sweep (read per-size cubes); else the config cache.
    cache_root = args.cache_root if args.cache_root is not None else cfg["cache_root"]
    splits_file = cfg["splits_file"]
    ply_element = cfg.get("ply_element", "params")
    label_field = cfg.get("label_field", "label_ch")

    out_root = args.out_root or os.path.join("predictions", dataset_name, "icp_euclidean")
    log = setup_logger(os.path.join(out_root, "predict_icp.log"))

    log.info("ICP+Euclidean baseline (Task 4)")
    log.info("dataset=%s raw_root=%s cache_root=%s", dataset_name, raw_root, cache_root)
    log.info("frozen map: logits[:,1]=(d-D0)/S0, scores=sigmoid(.)  D0=%.3f m S0=%.3f m", args.d0, args.s0)
    log.info("ICP: point-to-point, centered frame, max_corr=%.3f m, max_iter=%d", args.icp_corr, args.icp_iter)
    log.info("output dir: %s", out_root)
    log.info("")

    scenes = load_val_test_scenes(splits_file)
    os.makedirs(out_root, exist_ok=True)
    scene_to_sum_count = {}                               # scene_id -> sum(count_t1), for verify

    for split, scene_rel, scene_id in scenes:
        arrays, stats = predict_scene(
            scene_rel, scene_id, raw_root, cache_root,
            args.d0, args.s0, args.icp_corr, args.icp_iter, ply_element, label_field, dataset_name)
        scene_to_sum_count[scene_id] = stats["sum_count_t1"]

        # Write the per-scene Section 8 npz.
        out_path = os.path.join(out_root, scene_id + ".npz")
        np.savez(
            out_path,
            scores=arrays["scores"], logits=arrays["logits"], labels=arrays["labels"],
            coords=arrays["coords"], cube_id=arrays["cube_id"],
            scene_id=np.array(arrays["scene_id"]), dataset=np.array(arrays["dataset"]))

        # Internal consistency: N from npz must equal cache sum(count_t1).
        n_ok = "OK" if stats["num_points"] == stats["sum_count_t1"] else "MISMATCH"
        log.info(
            "%-4s %-8s N=%d (cache %d %s) cubes=%d chg=%.3f | ICP fit med=%.3f min=%.3f | "
            "F1@0.5=%.3f P=%.3f R=%.3f",
            split, scene_id, stats["num_points"], stats["sum_count_t1"], n_ok,
            stats["num_cubes"], stats["change_ratio"],
            stats["icp_fitness_median"], stats["icp_fitness_min"],
            stats["f1_at_0p5"], stats["precision_at_0p5"], stats["recall_at_0p5"])

    log.info("")
    log.info("Verifying Section 8 schema and N == sum(count_t1) on all written files...")
    n_files = verify_outputs(out_root, scene_to_sum_count, dataset_name)
    log.info("Schema + N checks passed for %d scene files.", n_files)
    log.info("Done.")


if __name__ == "__main__":
    main()
