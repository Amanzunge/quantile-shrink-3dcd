"""
Discrimination sanity check for saved predictions (any model, any dataset). This is
the gate to run BEFORE calibration (Task 7): if the scores do not rank change above
no-change, no threshold method can help, and the two_moment study would be meaningless.
It is also the check that exposed the Task 5 NN-propagation bug (saved AUC 0.57 while
the model's own AUC was 0.91).

For every predictions/<dataset>/<model>/<scene>.npz it reports:
  - AUC          ROC-AUC, the threshold-free discrimination signal (0.5 = random).
  - F1@0.5       F1 of the change class at the fixed 0.5 threshold.
  - bestG        F1 at the best SINGLE GLOBAL threshold (oracle for one dataset-wide tau).
  - oracle       per-cube ORACLE F1: each cube gets its own best threshold. This is an
                 UPPER BOUND for any per-cube calibration, including two_moment. Reported
                 two ways: pooled over all points, and the mean of per-cube best F1.

Torch-free (numpy only), so it runs in the CPU project env. Prints a table per model and
writes results/<dataset>_discrimination.csv.

  python src/diagnose_predictions.py --config configs/urb3dcd_v2_ld.yaml
"""

import os
import glob
import json
import csv
import argparse
import numpy as np
import yaml


def roc_auc(scores, labels):
    """ROC-AUC via Mann-Whitney U with average ranks for ties (sklearn-equivalent)."""
    n_pos = int(np.sum(labels == 1))
    n_neg = int(len(labels) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")                              # AUC undefined for a one-class scene
    order = np.argsort(scores, kind="mergesort")         # stable sort by score ascending
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)         # 1-based ranks
    s_sorted = scores[order]
    i = 0
    while i < len(s_sorted):                             # average the ranks of tied scores
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def f1_from_pred(pred, labels):
    """F1 of the change class given a boolean prediction and binary labels."""
    tp = int(np.sum(pred & (labels == 1)))
    fp = int(np.sum(pred & (labels == 0)))
    fn = int(np.sum((~pred) & (labels == 1)))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0


def best_global_f1(scores, labels, taus):
    """Best F1 over a single global threshold swept across taus."""
    best_f1, best_tau = 0.0, 0.5
    for tau in taus:
        f1 = f1_from_pred(scores > tau, labels)
        if f1 > best_f1:
            best_f1, best_tau = f1, float(tau)
    return best_f1, best_tau


def percube_oracle_f1(scores, labels, cube_id, taus):
    """
    Per-cube oracle: give every cube its own F1-optimal threshold (using labels), the
    upper bound for per-cube calibration. Returns (pooled F1 under those per-cube
    thresholds, mean of per-cube best F1 over cubes that contain at least one change).
    """
    pred = np.zeros(len(labels), dtype=bool)
    per_cube = []
    for cube in np.unique(cube_id):
        mask = cube_id == cube
        labels_c = labels[mask]
        scores_c = scores[mask]
        if labels_c.sum() == 0:
            pred[mask] = False                           # no change in cube -> predict none, no FP
            continue
        best_f1, best_pred = -1.0, None
        for tau in taus:
            this_pred = scores_c > tau
            f1 = f1_from_pred(this_pred, labels_c)
            if f1 > best_f1:
                best_f1, best_pred = f1, this_pred
        pred[mask] = best_pred
        per_cube.append(best_f1)
    pooled = f1_from_pred(pred, labels)
    mean_cube = float(np.mean(per_cube)) if per_cube else float("nan")
    return pooled, mean_cube


def diagnose_model(pred_dir, split_of, taus):
    """Build the per-scene + pooled rows for one model directory."""
    rows = []
    pool = {}
    for path in sorted(glob.glob(os.path.join(pred_dir, "*.npz"))):
        d = np.load(path, allow_pickle=True)
        scene = str(d["scene_id"])
        split = split_of.get(scene, "?")
        labels = d["labels"]
        scores = d["scores"]
        cube_id = d["cube_id"]
        bg_f1, bg_tau = best_global_f1(scores, labels, taus)
        orc_pool, orc_cube = percube_oracle_f1(scores, labels, cube_id, taus)
        rows.append({
            "scene": scene, "split": split, "n": len(labels),
            "change_ratio": round(float((labels == 1).mean()), 4),
            "auc": round(roc_auc(scores, labels), 4),
            "f1_at_0.5": round(f1_from_pred(scores > 0.5, labels), 4),
            "best_global_f1": round(bg_f1, 4), "best_global_tau": round(bg_tau, 2),
            "percube_oracle_pooled": round(orc_pool, 4),
            "percube_oracle_meancube": round(orc_cube, 4),
        })
        pool.setdefault(split, []).append((labels, scores))
    # Pooled rows per split (no per-cube oracle: cube ids are not unique across scenes).
    for split in ["val", "test"]:
        if split not in pool:
            continue
        labels = np.concatenate([x[0] for x in pool[split]])
        scores = np.concatenate([x[1] for x in pool[split]])
        bg_f1, bg_tau = best_global_f1(scores, labels, taus)
        rows.append({
            "scene": "POOLED", "split": split, "n": len(labels),
            "change_ratio": round(float((labels == 1).mean()), 4),
            "auc": round(roc_auc(scores, labels), 4),
            "f1_at_0.5": round(f1_from_pred(scores > 0.5, labels), 4),
            "best_global_f1": round(bg_f1, 4), "best_global_tau": round(bg_tau, 2),
            "percube_oracle_pooled": "", "percube_oracle_meancube": "",
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description="Discrimination sanity check for saved predictions.")
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
    taus = np.linspace(0.02, 0.98, 49)                   # threshold grid for the F1 sweeps

    fields = ["model", "scene", "split", "n", "change_ratio", "auc", "f1_at_0.5",
              "best_global_f1", "best_global_tau", "percube_oracle_pooled", "percube_oracle_meancube"]
    all_rows = []
    for model in models:
        rows = diagnose_model(os.path.join(pred_root, model), split_of, taus)
        print("\n== %s ==" % model)
        print("%-8s %-4s %-6s %-6s %-6s %-6s %-7s %-16s" %
              ("scene", "spl", "chg", "AUC", "F1@.5", "bestG", "bestTau", "oracle(pool/cube)"))
        for r in rows:
            op = r["percube_oracle_pooled"]
            tail = "" if op == "" else ("%.3f / %.3f" % (op, r["percube_oracle_meancube"]))
            print("%-8s %-4s %-6.3f %-6.3f %-6.3f %-6.3f %-7.2f %s" %
                  (r["scene"], r["split"], r["change_ratio"], r["auc"], r["f1_at_0.5"],
                   r["best_global_f1"], r["best_global_tau"], tail))
            out = {"model": model}
            out.update(r)
            all_rows.append(out)

    os.makedirs("results", exist_ok=True)
    out_csv = os.path.join("results", dataset + "_discrimination.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)
    print("\nwrote", out_csv)


if __name__ == "__main__":
    main()
