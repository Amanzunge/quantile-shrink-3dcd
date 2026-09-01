"""
calibrate_otsu.py - Per-cube Otsu baseline (pre-submission reviewer baseline).

The natural reviewer question about per-cube calibration is "why not just run
Otsu inside every cube?". This script answers it with the real thing: for every
cube, the classic Otsu threshold (maximum between-class variance on a 256-bin
histogram of the cube's scores), applied with the same strict score > tau
decision as every other method. It is fully unsupervised and parameter-free:
NOTHING is fit on val, which is exactly the baseline a reviewer would propose.

A cube whose histogram is degenerate (all scores in one bin, so no cut exists)
falls back to the scene-pooled Otsu threshold.

Outputs (frozen artefacts are never touched):
  calibration_v2/<dataset>/<model>/otsu_cube.json   same schema as the other methods
  results/otsu_baseline_table.csv                   same columns as main_table.csv, upserted

Torch-free, local CPU:  python src/calibrate_otsu.py --config configs/hkcd.yaml
"""

import os
import json
import csv
import argparse
import numpy as np
import yaml

# Reuse the frozen Task 7 machinery so F1 conventions and the JSON schema are
# identical by construction.
from calibrate import r, assemble
from calibrate_v2 import load_split_scores

# Score histogram resolution. Scores live in [0, 1], so 256 bins give the same
# granularity as the classic 8-bit-image Otsu.
BINS = 256

METHOD = "otsu_cube"


def otsu_cut(hist):
    """Vectorized Otsu over the rows of a (K, BINS) histogram. For each row, pick
    the cut AFTER bin t (class 0 = bins 0..t) that maximizes the between-class
    variance w0*w1*(mu0-mu1)^2. Returns (tau, degenerate): tau is the bin edge
    (t+1)/BINS, so the strict score > tau decision sends bins 0..t to no-change;
    degenerate marks rows where no cut separates anything (all mass in one bin)."""
    w = hist.astype(np.float64)
    centers = (np.arange(BINS) + 0.5) / BINS
    w0 = np.cumsum(w, axis=1)                        # class-0 mass for every cut
    m0 = np.cumsum(w * centers[None, :], axis=1)     # class-0 first moment
    w1 = w0[:, -1:] - w0                             # class-1 mass
    m1 = m0[:, -1:] - m0
    mu0 = m0 / np.maximum(w0, 1e-12)                 # class means, guarded for empty classes
    mu1 = m1 / np.maximum(w1, 1e-12)
    sigma_b = w0 * w1 * (mu0 - mu1) ** 2
    sigma_b[:, -1] = 0.0                             # cutting after the last bin is no cut
    best = np.argmax(sigma_b, axis=1)
    tau = (best + 1.0) / BINS
    degenerate = sigma_b[np.arange(len(w)), best] <= 0.0
    return tau, degenerate


def otsu_scene(rec):
    """Per-cube Otsu thresholds for one scene. Degenerate cubes fall back to the
    scene-pooled Otsu threshold. Returns (tau per cube, point->cube index, n_fallback)."""
    scores = rec["scores"]
    cubes, inv = np.unique(rec["cube_id"], return_inverse=True)
    K = len(cubes)
    bin_of = np.minimum((scores * BINS).astype(np.int64), BINS - 1)  # score 1.0 goes in the last bin
    # One flat bincount builds all K histograms at once (cube i occupies slots i*BINS..).
    hist = np.bincount(inv * BINS + bin_of, minlength=K * BINS).reshape(K, BINS)
    tau, degenerate = otsu_cut(hist)
    scene_tau, scene_degen = otsu_cut(hist.sum(axis=0, keepdims=True))
    fallback = 0.5 if scene_degen[0] else float(scene_tau[0])
    tau = np.where(degenerate, fallback, tau)
    return tau, inv, int(degenerate.sum())


def main():
    parser = argparse.ArgumentParser(description="Per-cube Otsu baseline on saved predictions.")
    parser.add_argument("--config", required=True, help="dataset YAML in configs/")
    parser.add_argument("--models", nargs="*", default=None,
                        help="model names under predictions/<dataset>/ (default: all found)")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    dataset = cfg["dataset_name"]
    splits = json.load(open(cfg["splits_file"]))
    split_of = {}
    for split in ["train", "val", "test"]:
        for rel in splits.get(split, []):
            split_of[rel.split("/")[-1]] = split

    pred_root = os.path.join("predictions", dataset)
    models = args.models or sorted(
        m for m in os.listdir(pred_root) if os.path.isdir(os.path.join(pred_root, m)))

    # Reference numbers from the frozen main table, for the printed comparison.
    reference = {}
    main_path = os.path.join("results", "main_table.csv")
    if os.path.exists(main_path):
        with open(main_path, newline="") as f:
            for row in csv.DictReader(f):
                if row["dataset"] == dataset and row["method"] in ("f1_optimal", "quantile_shrink"):
                    reference[(row["model"], row["method"])] = float(row["test_mean_f1"])

    table_rows = []
    for model in models:
        val, test = load_split_scores(os.path.join(pred_root, model), split_of)
        print("\n== %s ==  (val %d scenes, test %d scenes)" % (model, len(val), len(test)))

        # Per-scene Otsu structures, computed once and reused by predict/tau/report.
        tau_by_scene, inv_by_scene, nfb_by_scene = {}, {}, {}
        for rec in (val + test):
            tau, inv, n_fb = otsu_scene(rec)
            tau_by_scene[rec["scene"]] = tau
            inv_by_scene[rec["scene"]] = inv
            nfb_by_scene[rec["scene"]] = n_fb

        predict_fn = lambda rec: rec["scores"] > tau_by_scene[rec["scene"]][inv_by_scene[rec["scene"]]]
        tau_of_scene = lambda rec: [r(v) for v in tau_by_scene[rec["scene"]]]
        fitted = {"bins": BINS, "fitted": "none (parameter-free, val unused)",
                  "fallback": "scene-pooled Otsu where the cube histogram is degenerate"}
        result = assemble(model, METHOD, dataset, val, test, predict_fn, tau_of_scene, fitted)
        for rec in (val + test):
            result["per_scene"][rec["scene"]]["n_cubes"] = int(len(tau_by_scene[rec["scene"]]))
            result["per_scene"][rec["scene"]]["n_fallback"] = nfb_by_scene[rec["scene"]]

        out_dir = os.path.join("calibration_v2", dataset, model)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, METHOD + ".json"), "w") as f:
            json.dump(result, f, indent=2)

        s = result["summary"]
        table_rows.append({
            "dataset": dataset, "model": model, "method": METHOD,
            "val_mean_p": s["val_mean_p"], "val_mean_r": s["val_mean_r"], "val_mean_f1": s["val_mean_f1"],
            "test_mean_p": s["test_mean_p"], "test_mean_r": s["test_mean_r"], "test_mean_f1": s["test_mean_f1"],
            "val_pool_p": s["val_pooled"]["precision"], "val_pool_r": s["val_pooled"]["recall"],
            "val_pool_f1": s["val_pooled"]["f1"],
            "test_pool_p": s["test_pooled"]["precision"], "test_pool_r": s["test_pooled"]["recall"],
            "test_pool_f1": s["test_pooled"]["f1"],
        })
        ref_fo = reference.get((model, "f1_optimal"), float("nan"))
        ref_qs = reference.get((model, "quantile_shrink"), float("nan"))
        print("  otsu_cube test F1 %.4f   vs f1_optimal %+.4f   vs quantile_shrink %+.4f"
              % (s["test_mean_f1"], s["test_mean_f1"] - ref_fo, s["test_mean_f1"] - ref_qs))

    # UPSERT into the otsu baseline table (same discipline as main_table.csv).
    os.makedirs("results", exist_ok=True)
    table_fields = ["dataset", "model", "method", "val_mean_p", "val_mean_r", "val_mean_f1",
                    "test_mean_p", "test_mean_r", "test_mean_f1", "val_pool_p", "val_pool_r",
                    "val_pool_f1", "test_pool_p", "test_pool_r", "test_pool_f1"]
    out_path = os.path.join("results", "otsu_baseline_table.csv")
    kept = []
    if os.path.exists(out_path):
        with open(out_path, newline="") as f:
            kept = [row for row in csv.DictReader(f) if row.get("dataset") != dataset]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=table_fields, extrasaction="ignore")
        w.writeheader()
        for row in kept + table_rows:
            w.writerow(row)

    assert len(table_rows) == len(models), "missing model rows"
    print("\nupserted %d %s rows into results/otsu_baseline_table.csv (kept %d other rows)"
          % (len(table_rows), dataset, len(kept)))
    print("wrote calibration_v2/%s/<model>/%s.json" % (dataset, METHOD))


if __name__ == "__main__":
    main()
