"""
Task 12: build results/ablation_fps.csv and results/ablation_cube.csv from the
ablation predictions under ablations/<dataset>/<sweep>/<value>/<model>/.

This reuses the LOCKED calibration math (src/calibrate.py + src/calibrate_v2.py) and
discrimination math (src/diagnose_predictions.py) UNCHANGED, so every ablation number
is computed exactly the way the paper tables are (mean per-scene change-F1, score<=0.5
no-change subset, the same C/ALPHA/K grids as calibration_v2). It NEVER writes
predictions/ or results/main_table.csv -- ablation results live only in
results/ablation_*.csv.

Since the 2026-07-14 paper-layer decision the per-cube method reported here is the
HEADLINE quantile_shrink (calibrate_v2), not the removed two_moment ablation.

The MAIN point of each sweep (cube 50 m / indoor 1.5 m; FPS 1024 LD / 4096 MS) is
NOT re-run: it is read straight from the main predictions/<dataset>/<model>/ so the
sweep curve has its anchor without retraining.

Per ablation point (dataset, swept value, model) the row carries:
  pooled_test_auc       gate (threshold-free discrimination; <0.6 = the model failed here)
  f1_optimal_test       one global tuned tau (val-fit), test mean per-scene change-F1
  quantile_shrink_test  per-cube (c,alpha,k) (val-fit), test mean per-scene change-F1  (OURS)
  quantile_shrink_gain  quantile_shrink_test - f1_optimal_test  (the realised per-cube benefit)
  oracle_ceiling        per-cube oracle (test, mean per-scene pooled): the realisable upper bound
  oracle_headroom       oracle_ceiling - f1_optimal_test   (signal that EXISTS per cube)
  mean_shrink_w_test    mean over test cubes of w = n/(n+k); low = the soft fallback is
                        pulling thresholds toward the scene-level quantile
  c, alpha, k           the fitted quantile_shrink scalars
plus the model-free per-cube GEOMETRY for that point:
  median_cubes_per_scene, total_kept_cubes, pct_zero_change_cubes, cube_drop_pct
The cube-size sweep MOVES this geometry (smaller cubes -> more, sparser, emptier cubes);
the FPS sweep does NOT (FPS changes only the model input resolution, not the cube
partition), so its geometry columns repeat the MAIN cube grid and are flagged constant.

Run:
  python src/build_ablation_tables.py --sweep cube
  python src/build_ablation_tables.py --sweep fps
  python src/build_ablation_tables.py --sweep both   (default)
"""

import os
import csv
import json
import argparse
import numpy as np

# Locked calibration math (identical to the paper tables).
from calibrate import sweep_global_tau, mean_perscene_f1
from calibrate_v2 import (load_split_scores, precompute_scene, fit_quantile_shrink,
                          tau_quantile_shrink, shrink_weight, ALPHA_GRID)
# Locked discrimination math.
from diagnose_predictions import roc_auc, percube_oracle_f1


# ---------------------------------------------------------------------------
# Sweep definitions (CLAUDE Section 4 / 5 + the Task 12 locked policy in 5.4).
# value tokens are the directory names used under ablations/<ds>/<sweep>/<value>/
# and (for cube) the data/cache/<ds>_cube<value> per-size cache suffix. MAIN_VALUE
# is the anchor read from predictions/ (not retrained).
# ---------------------------------------------------------------------------

# FPS sweep: PRIMARY benchmark only (LD + MS); icp excluded (no FPS step).
FPS_SWEEP = {
    "urb3dcd_v2_ld": {"values": ["256", "512", "1024"], "main": "1024",
                      "models": ["siamese_pointnet", "siamese_pointnet2",
                                 "siamese_kpconv", "siamgcn"]},
    "urb3dcd_v2_ms": {"values": ["256", "512", "1024", "2048", "4096"], "main": "4096",
                      "models": ["siamese_pointnet", "siamese_pointnet2",
                                 "siamese_kpconv", "siamgcn", "randla"]},
}

# Cube-size sweep: all models + icp (icp outdoor only; indoor is deep-only, Section 4).
OUTDOOR_DEEP = ["siamese_pointnet", "siamese_pointnet2", "siamese_kpconv", "siamgcn"]
CUBE_SWEEP = {
    "urb3dcd_v2_ld": {"values": ["25", "50", "75"], "main": "50",
                      "models": OUTDOOR_DEEP + ["icp_euclidean"]},
    "urb3dcd_v2_ms": {"values": ["25", "50", "75"], "main": "50",
                      "models": OUTDOOR_DEEP + ["randla", "icp_euclidean"]},
    "hkcd":          {"values": ["25", "50", "75"], "main": "50",
                      "models": OUTDOOR_DEEP + ["randla", "icp_euclidean"]},
    "indoorcd":      {"values": ["1.0", "1.5", "2.0"], "main": "1.5",
                      "models": OUTDOOR_DEEP + ["randla"]},
}

# Same threshold grid the diagnostic uses for the per-cube oracle sweep.
ORACLE_TAUS = np.linspace(0.02, 0.98, 49)


def split_of_dataset(config_path):
    """scene_id -> split ('train'|'val'|'test') from the frozen splits of one dataset."""
    import yaml
    cfg = yaml.safe_load(open(config_path))
    splits = json.load(open(cfg["splits_file"]))
    split_of = {}
    for split in ["train", "val", "test"]:
        for rel in splits.get(split, []):
            split_of[rel.split("/")[-1]] = split
    return split_of, cfg["dataset_name"]


def cache_root_for(dataset, sweep, value, main_value):
    """Cache dir holding this point's cube grid (for the model-free GEOMETRY only).
    FPS never changes the partition, so it always reads the main cache."""
    if sweep == "fps":
        return os.path.join("data", "cache", dataset)
    if value == main_value:                                   # cube-size MAIN point
        return os.path.join("data", "cache", dataset)
    return os.path.join("data", "cache", dataset + "_cube" + value)


def pred_dir_for(dataset, sweep, value, main_value, model):
    """Where this point's per-scene npz live: the MAIN value reuses predictions/, the
    swept values live under ablations/."""
    if value == main_value:
        return os.path.join("predictions", dataset, model)
    return os.path.join("ablations", dataset, sweep, value, model)


def geometry(cache_root, dataset, value):
    """Model-free per-cube geometry for one cube grid: median kept cubes per scene,
    total kept cubes, % of kept cubes with zero changed t1 points, and the
    min_points=256 drop rate if a matching inventory CSV exists."""
    manifest_path = os.path.join(cache_root, "manifest.json")
    if not os.path.exists(manifest_path):
        return {}
    manifest = json.load(open(manifest_path))
    kept = np.array([int(s["cubes_kept"]) for s in manifest["scenes"]])
    median_cubes = float(np.median(kept))
    total_kept = int(kept.sum())
    # Zero-change cube %: load each scene cache's per-cube n_changed (kept cubes only).
    n_changed = []
    for s in manifest["scenes"]:
        cache = np.load(os.path.join(cache_root, s["scene_id"] + ".npz"), allow_pickle=False)
        n_changed.append(cache["n_changed"])
    n_changed = np.concatenate(n_changed)
    zero_pct = 100.0 * float((n_changed == 0).mean())
    # Drop %: needs cubes_occupied, which only the inventory CSV records.
    drop_pct = ""
    inv = os.path.join("results", inventory_name(dataset, value))
    if os.path.exists(inv):
        occ = dropped = 0
        for row in csv.DictReader(open(inv)):
            occ += int(row["cubes_occupied"])
            dropped += int(row["cubes_dropped"])
        drop_pct = round(100.0 * dropped / occ, 1) if occ else ""
    return {"median_cubes_per_scene": round(median_cubes, 1),
            "total_kept_cubes": total_kept,
            "pct_zero_change_cubes": round(zero_pct, 1),
            "cube_drop_pct": drop_pct}


def inventory_name(dataset, value):
    """Inventory CSV name for a cube grid: main grid keeps the Task-1/9/11 name; a swept
    size uses results/<ds>_cube<value>_inventory.csv (written by inspect_dataset --cube-size)."""
    main = {"urb3dcd_v2_ld": "50", "urb3dcd_v2_ms": "50", "hkcd": "50", "indoorcd": "1.5"}
    if value == main.get(dataset):
        return dataset + "_inventory.csv"
    return dataset + "_cube" + value + "_inventory.csv"


def fit_point(val, test):
    """Run f1_optimal + quantile_shrink on one point's loaded val/test scenes and return the
    full metric dict. EXACT same pipeline as calibrate.run_f1_optimal / calibrate_v2's
    quantile_shrink and diagnose (AUC, per-cube oracle) -- just without the per-method JSON."""
    # f1_optimal: one global tau swept on val (mean per-scene F1), the gain baseline.
    f1opt_tau, _ = sweep_global_tau(val, lambda rec: rec["scores"])
    f1opt_test = mean_perscene_f1(test, lambda rec, t=f1opt_tau: rec["scores"] > t)

    # quantile_shrink: fit (c, alpha, k) once on val via the exact calibrate_v2 machinery.
    pre_by_scene = {rec["scene"]: precompute_scene(rec) for rec in (val + test)}
    c, ai, k, _ = fit_quantile_shrink([pre_by_scene[rec["scene"]] for rec in val])

    # Reported test F1 from plain per-point predictions, never the fast search path.
    def qs_pred(rec):
        pre = pre_by_scene[rec["scene"]]
        return rec["scores"] > tau_quantile_shrink(pre, c, ai, k)[pre["inv"]]
    qs_test = mean_perscene_f1(test, qs_pred)

    # Per-cube oracle ceiling on test (mean per-scene pooled): the realisable upper bound,
    # SAME aggregation as f1_optimal/quantile_shrink (build_scope_limit_table gotcha).
    orc = []
    for rec in test:
        pooled, _ = percube_oracle_f1(rec["scores"], rec["labels"], rec["cube_id"], ORACLE_TAUS)
        orc.append(pooled)
    oracle_ceiling = float(np.mean(orc)) if orc else float("nan")

    # Pooled-test AUC (the gate value the memories quote).
    test_scores = np.concatenate([rec["scores"] for rec in test])
    test_labels = np.concatenate([rec["labels"] for rec in test])
    pooled_auc = roc_auc(test_scores, test_labels)

    # Soft-fallback pressure: mean shrink weight over test cubes (replaces the old MIN_NC
    # fallback %; low w = thresholds lean on the scene-level quantile).
    w_all = [shrink_weight(pre_by_scene[rec["scene"]]["n_nc"], k) for rec in test]
    mean_w = float(np.mean(np.concatenate(w_all))) if w_all else float("nan")

    return {
        "pooled_test_auc": round(pooled_auc, 4),
        "f1_optimal_test": round(f1opt_test, 4),
        "quantile_shrink_test": round(qs_test, 4),
        "quantile_shrink_gain": round(qs_test - f1opt_test, 4),
        "oracle_ceiling": round(oracle_ceiling, 4),
        "oracle_headroom": round(oracle_ceiling - f1opt_test, 4),
        "mean_shrink_w_test": round(mean_w, 3),
        "c": round(c, 3), "alpha": float(ALPHA_GRID[ai]), "k": k,
        "f1_optimal_tau": round(f1opt_tau, 3),
        "n_val_scenes": len(val), "n_test_scenes": len(test),
    }


COLS = ["dataset", "sweep", "value", "is_main", "model",
        "pooled_test_auc", "f1_optimal_test", "quantile_shrink_test", "quantile_shrink_gain",
        "oracle_ceiling", "oracle_headroom", "mean_shrink_w_test", "c", "alpha", "k",
        "f1_optimal_tau", "median_cubes_per_scene", "total_kept_cubes",
        "pct_zero_change_cubes", "cube_drop_pct", "n_val_scenes", "n_test_scenes"]


def build(sweep):
    """Build one ablation table; returns the rows (also written to results/ablation_<sweep>.csv)."""
    plan = FPS_SWEEP if sweep == "fps" else CUBE_SWEEP
    rows = []
    for dataset, spec in plan.items():
        config = os.path.join("configs", dataset + ".yaml")
        split_of, _ = split_of_dataset(config)
        for value in spec["values"]:
            geo = geometry(cache_root_for(dataset, sweep, value, spec["main"]), dataset, value)
            for model in spec["models"]:
                pred_dir = pred_dir_for(dataset, sweep, value, spec["main"], model)
                if not (os.path.isdir(pred_dir) and
                        any(f.endswith(".npz") for f in os.listdir(pred_dir))):
                    continue                                  # not produced yet (Colab pending)
                val, test = load_split_scores(pred_dir, split_of)
                if not test:                                  # nothing usable
                    continue
                row = {"dataset": dataset, "sweep": sweep, "value": value,
                       "is_main": int(value == spec["main"]), "model": model}
                row.update(fit_point(val, test))
                row.update(geo)
                rows.append(row)
                print("  %-14s %-5s %-18s gain=%+.4f head=%+.4f auc=%.3f" % (
                    dataset, value, model, row["quantile_shrink_gain"],
                    row["oracle_headroom"], row["pooled_test_auc"]))

    os.makedirs("results", exist_ok=True)
    out = os.path.join("results", "ablation_" + sweep + ".csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print("wrote %s with %d rows" % (out, len(rows)))
    return rows


def build_geometry():
    """Model-free per-cube GEOMETRY for every (dataset, cube size), independent of any model
    or training. This is the continuous scope-limit boundary: as cubes shrink, cubes/scene
    rises, the min_points drop rate rises, and the share of zero-change cubes rises -- walking
    the outdoor datasets toward the IndoorCD room-scale floor. Written to
    results/ablation_cube_geometry.csv; completable with zero predictions."""
    cols = ["dataset", "value", "is_main", "cube_extent_m", "median_cubes_per_scene",
            "total_kept_cubes", "pct_zero_change_cubes", "cube_drop_pct"]
    rows = []
    print("\n== cube geometry (model-free boundary) ==")
    print("  %-14s %-5s %-9s %-9s %-7s %-6s" % ("dataset", "size", "med_cubes", "tot_kept", "zero%", "drop%"))
    for dataset, spec in CUBE_SWEEP.items():
        for value in spec["values"]:
            geo = geometry(cache_root_for(dataset, "cube", value, spec["main"]), dataset, value)
            if not geo:
                continue                                       # cache for this size not built yet
            row = {"dataset": dataset, "value": value, "is_main": int(value == spec["main"]),
                   "cube_extent_m": value}
            row.update(geo)
            rows.append(row)
            print("  %-14s %-5s %-9s %-9s %-7s %-6s" % (
                dataset, value, row["median_cubes_per_scene"], row["total_kept_cubes"],
                row["pct_zero_change_cubes"], row["cube_drop_pct"] or "-"))
    os.makedirs("results", exist_ok=True)
    with open(os.path.join("results", "ablation_cube_geometry.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print("wrote results/ablation_cube_geometry.csv with %d rows" % len(rows))
    return rows


def main():
    parser = argparse.ArgumentParser(description="Build Task 12 ablation tables from saved predictions.")
    parser.add_argument("--sweep", choices=["fps", "cube", "geometry", "both"], default="both")
    args = parser.parse_args()
    if args.sweep in ("geometry", "both", "cube"):
        build_geometry()                                       # always refresh the model-free boundary
    sweeps = []
    if args.sweep in ("fps", "both"):
        sweeps.append("fps")
    if args.sweep in ("cube", "both"):
        sweeps.append("cube")
    for sweep in sweeps:
        print("\n== %s sweep ==" % sweep)
        build(sweep)


if __name__ == "__main__":
    main()
