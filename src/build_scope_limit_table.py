"""
Task 11 deliverable: results/scope_limit_table.csv.

Joins, per dataset and deep model, the model-free per-cube GEOMETRY (how populated
each scene's cube grid is) with the two-moment calibration OUTCOME, so the IndoorCD
room-scale floor can be read directly against the outdoor datasets (LD/MS/HKCD).

Geometry (per dataset, model-independent):
  median_cubes_per_scene   median kept cubes per scene (from the inventory CSV)
  cube_drop_pct            dropped / occupied cubes, the min_points=256 rule (inventory)
  pct_zero_change_cubes    % of kept cubes with ZERO changed t1 points (from the cache)

Calibration (per dataset x deep model, TEST split, mean per-scene change-F1):
  test_f1_optimal / test_two_moment / two_moment_gain
  oracle_ceiling / oracle_headroom  per-cube oracle (best per-cube tau) and its gap
                                    over the tuned global threshold (the realisable upper bound)
  fallback_pct_test   % of test cubes that fell back to the global tau (no-change subset
                      < MIN_NC=10). NOTE: this tracks model OVER-PREDICTION, not scale.

Run:  python src/build_scope_limit_table.py
"""

import os
import csv
import json
import numpy as np

# Deep models only (Section 4: IndoorCD is deep-only; icp excluded everywhere here).
DATASETS = ["urb3dcd_v2_ld", "urb3dcd_v2_ms", "hkcd", "indoorcd"]
DEEP = ["siamese_pointnet", "siamese_pointnet2", "siamese_kpconv", "siamgcn", "randla"]


def inventory_geometry(ds):
    """Median kept cubes per scene and the overall cube drop rate (%)."""
    rows = list(csv.DictReader(open("results/" + ds + "_inventory.csv")))
    kept = np.array([int(r["cubes_kept"]) for r in rows])
    occupied = np.array([int(r["cubes_occupied"]) for r in rows])
    dropped = np.array([int(r["cubes_dropped"]) for r in rows])
    return float(np.median(kept)), 100.0 * dropped.sum() / occupied.sum()


def zero_change_pct(ds):
    """% of kept cubes (all splits) whose t1 cloud holds zero changed points."""
    manifest = json.load(open("data/cache/" + ds + "/manifest.json"))
    n_changed = []
    for scene in manifest["scenes"]:
        cache = np.load("data/cache/" + ds + "/" + scene["scene_id"] + ".npz", allow_pickle=False)
        n_changed.append(cache["n_changed"])
    n_changed = np.concatenate(n_changed)
    return 100.0 * float((n_changed == 0).mean())


def main_table_f1(ds, model, method):
    """test_mean_f1 for one (dataset, model, method) row, or None if absent."""
    for r in csv.DictReader(open("results/main_table.csv")):
        if r["dataset"] == ds and r["model"] == model and r["method"] == method:
            return float(r["test_mean_f1"])
    return None


def fallback_pct_test(ds, model):
    """% of TEST cubes that hit the MIN_NC=10 fallback in the two_moment fit."""
    jf = "calibration/" + ds + "/" + model + "/two_moment.json"
    if not os.path.exists(jf):
        return None
    per_scene = json.load(open(jf))["per_scene"]
    n_fallback = n_cubes = 0
    for v in per_scene.values():
        if v.get("split") == "test":
            n_fallback += v.get("n_fallback", 0)
            n_cubes += v.get("n_cubes", 0)
    return 100.0 * n_fallback / n_cubes if n_cubes else None


def oracle_ceiling(ds, model):
    """
    Per-cube oracle ceiling = mean over TEST scenes of percube_oracle_pooled (each scene's
    F1 when every cube uses its own best tau, pooled over the scene's points). This is the
    SAME aggregation as f1_optimal/two_moment (mean per-scene, pooled within a scene), so the
    headroom (oracle - f1_optimal) is a fair realisable upper bound -- it matches the ceiling
    that calibrate.py reports. (percube_oracle_meancube averages F1 over cubes instead and is
    NOT comparable to the global-threshold per-scene F1.)
    """
    path = "results/" + ds + "_discrimination.csv"
    if not os.path.exists(path):
        return None
    vals = []
    for r in csv.DictReader(open(path)):
        if (r.get("model") == model and r.get("split") == "test"
                and str(r.get("scene", "")).upper() != "POOLED"):
            v = r.get("percube_oracle_pooled", "")
            if v not in ("", "nan"):
                vals.append(float(v))
    return float(np.mean(vals)) if vals else None


def main():
    out_rows = []
    for ds in DATASETS:
        median_cubes, drop_pct = inventory_geometry(ds)
        zchg = zero_change_pct(ds)
        for model in DEEP:
            f1_optimal = main_table_f1(ds, model, "f1_optimal")
            if f1_optimal is None:
                continue  # model not run on this dataset (e.g. randla excluded on LD)
            two_moment = main_table_f1(ds, model, "two_moment")
            oracle = oracle_ceiling(ds, model)
            fallback = fallback_pct_test(ds, model)
            out_rows.append({
                "dataset": ds,
                "model": model,
                "median_cubes_per_scene": round(median_cubes, 1),
                "cube_drop_pct": round(drop_pct, 1),
                "pct_zero_change_cubes": round(zchg, 1),
                "test_f1_optimal": round(f1_optimal, 4),
                "test_two_moment": round(two_moment, 4) if two_moment is not None else "",
                "two_moment_gain": round(two_moment - f1_optimal, 4) if two_moment is not None else "",
                "oracle_ceiling": round(oracle, 4) if oracle is not None else "",
                "oracle_headroom": round(oracle - f1_optimal, 4) if oracle is not None else "",
                "fallback_pct_test": round(fallback, 1) if fallback is not None else "",
            })

    cols = ["dataset", "model", "median_cubes_per_scene", "cube_drop_pct",
            "pct_zero_change_cubes", "test_f1_optimal", "test_two_moment",
            "two_moment_gain", "oracle_ceiling", "oracle_headroom", "fallback_pct_test"]
    with open("results/scope_limit_table.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(out_rows)
    print("wrote results/scope_limit_table.csv with", len(out_rows), "rows\n")

    # Pretty-print so the floor is readable at a glance.
    hdr = "{:<14} {:<18} {:>7} {:>6} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>6}".format(
        "dataset", "model", "cubes", "drop%", "zero%", "f1opt", "two_m", "gain", "oracle", "head", "fb%")
    print(hdr)
    print("-" * len(hdr))
    for r in out_rows:
        print("{:<14} {:<18} {:>7} {:>6} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>6}".format(
            r["dataset"], r["model"], r["median_cubes_per_scene"], r["cube_drop_pct"],
            r["pct_zero_change_cubes"], r["test_f1_optimal"], r["test_two_moment"],
            r["two_moment_gain"], r["oracle_ceiling"], r["oracle_headroom"], r["fallback_pct_test"]))


if __name__ == "__main__":
    main()
