"""
sensitivity.py - How sharply does quantile_shrink depend on its three fitted scalars?

Answers the obvious reviewer question: (c, alpha, k) are fitted on one or two
validation scenes, so is the test result an artefact of a finely tuned optimum?

For every model x dataset pair it re-walks the SAME 61 x 14 x 7 grid as
calibrate_v2.fit_quantile_shrink, scoring each candidate on val AND on test, then
reports how flat the val objective is and how much the test result moves across
the candidates that val cannot distinguish from the winner.

Torch-free, local CPU:  python src/sensitivity.py
Writes results/sensitivity_table.csv
"""

import os
import csv
import json
import numpy as np
import yaml

import calibrate_v2 as cv
from calibrate import sweep_global_tau

CONFIGS = ["configs/urb3dcd_v2_ld.yaml", "configs/urb3dcd_v2_ms.yaml",
           "configs/hkcd.yaml", "configs/indoorcd.yaml"]
BAND = 0.01          # val-F1 band that counts as "indistinguishable from the winner"


def grid_surfaces(pre_list):
    """Mean per-scene F1 for every (alpha, k, c) candidate over the given scenes.
    Returns an array shaped (n_alpha, n_k, n_c) - the same walk fit_quantile_shrink
    does, but keeping the whole surface instead of only its maximum."""
    out = np.zeros((len(cv.ALPHA_GRID), len(cv.K_GRID), len(cv.C_GRID)))
    for ai in range(len(cv.ALPHA_GRID)):
        for ki, k in enumerate(cv.K_GRID):
            f1_sum = np.zeros(len(cv.C_GRID))
            for pre in pre_list:
                w = cv.shrink_weight(pre["n_nc"], k)
                base = w * pre["q_by_alpha"][ai] + (1.0 - w) * pre["g_quant"][ai]
                rows = np.clip(cv.C_GRID[:, None] + base[None, :], 0.0, 1.0)
                f1_sum += cv.f1_batch(pre, rows)
            out[ai, ki] = f1_sum / len(pre_list)
    return out


def main():
    rows = []
    val_cube = {}        # (dataset, model) -> val surface, for the shared-default pass
    test_cube = {}
    global_f1 = {}       # (dataset, model) -> tuned global threshold test F1

    for config in CONFIGS:
        cfg = yaml.safe_load(open(config))
        dataset = cfg["dataset_name"]
        splits = json.load(open(cfg["splits_file"]))
        split_of = {}
        for split in ["train", "val", "test"]:
            for rel in splits.get(split, []):
                split_of[rel.split("/")[-1]] = split

        pred_root = os.path.join("predictions", dataset)
        models = sorted(m for m in os.listdir(pred_root)
                        if os.path.isdir(os.path.join(pred_root, m)))
        print("\n== %s ==" % dataset)

        for model in models:
            val, test = cv.load_split_scores(os.path.join(pred_root, model), split_of)
            pre = {rec["scene"]: cv.precompute_scene(rec) for rec in (val + test)}
            val_pre = [pre[rec["scene"]] for rec in val]
            test_pre = [pre[rec["scene"]] for rec in test]

            v = grid_surfaces(val_pre)          # (alpha, k, c) mean val F1
            t = grid_surfaces(test_pre)         # same candidates, scored on test
            val_cube[(dataset, model)] = v
            test_cube[(dataset, model)] = t

            # the deployed point: the val argmax, exactly as calibrate_v2 picks it
            ai, ki, ci = np.unravel_index(np.argmax(v), v.shape)
            # tuned global threshold, fitted on val and read off on test
            fb_tau, _ = sweep_global_tau(val, lambda rec: rec["scores"])
            gtest = float(np.mean([cv.f1_batch(p, np.full((1, p["K"]), fb_tau))[0]
                                   for p in test_pre]))
            global_f1[(dataset, model)] = gtest

            near = v >= v[ai, ki, ci] - BAND    # candidates val cannot separate
            t_near = t[near]
            rows.append({
                "dataset": dataset, "model": model,
                "c": cv.C_GRID[ci], "alpha": cv.ALPHA_GRID[ai], "k": cv.K_GRID[ki],
                "val_best": round(float(v[ai, ki, ci]), 4),
                "test_deployed": round(float(t[ai, ki, ci]), 4),
                "test_global": round(gtest, 4),
                "n_grid": int(v.size),
                "n_near": int(near.sum()),
                "frac_near": round(float(near.mean()), 4),
                "test_near_median": round(float(np.median(t_near)), 4),
                "test_near_p05": round(float(np.percentile(t_near, 5)), 4),
                "test_near_p95": round(float(np.percentile(t_near, 95)), 4),
                "test_near_min": round(float(t_near.min()), 4),
                "frac_near_beating_global": round(float((t_near >= gtest).mean()), 4),
                "test_shared": 0.0,          # filled in by the shared-default pass below
            })
            print("   %-20s val %.4f  test %.4f  near-optimal %d/%d  %.0f%% of them beat global"
                  % (model, v[ai, ki, ci], t[ai, ki, ci], near.sum(), v.size,
                     100 * (t_near >= gtest).mean()))

    # --- can two of the three scalars be shared across all 22 pairs? ---
    # For every (alpha, k) fixed globally, let c stay per-pair (it absorbs the
    # dataset-specific truncation gap), and score the result on test.
    keys = list(val_cube)
    best = None
    for ai in range(len(cv.ALPHA_GRID)):
        for ki in range(len(cv.K_GRID)):
            tot = 0.0
            for key in keys:
                ci = int(np.argmax(val_cube[key][ai, ki]))     # c still fitted on val
                tot += test_cube[key][ai, ki, ci]
            mean_test = tot / len(keys)
            if best is None or mean_test > best[0]:
                best = (mean_test, cv.ALPHA_GRID[ai], cv.K_GRID[ki])
    per_pair = np.mean([r["test_deployed"] for r in rows])
    print("\nshared (alpha, k) = (%.3f, %g): mean test F1 %.4f  vs per-pair %.4f  (cost %.4f)"
          % (best[1], best[2], best[0], per_pair, per_pair - best[0]))
    print("mean test F1 of the tuned global threshold: %.4f" % np.mean(list(global_f1.values())))

    # Per-pair detail for the shared setting, so the text can quote how many pairs
    # keep their advantage when only c is refitted.
    ai = int(np.where(cv.ALPHA_GRID == best[1])[0][0])
    ki = int(np.where(cv.K_GRID == best[2])[0][0])
    print("\n   shared-(alpha,k) per pair (c still fitted on val):")
    for key in keys:
        ci = int(np.argmax(val_cube[key][ai, ki]))
        shared = float(test_cube[key][ai, ki, ci])
        row = [r for r in rows if (r["dataset"], r["model"]) == key][0]
        row["test_shared"] = round(shared, 4)
        print("      %-16s %-20s shared %.4f  own %.4f  global %.4f"
              % (key[0], key[1], shared, row["test_deployed"], global_f1[key]))
    print("\n   shared setting: alpha = %.3f, k = %g" % (best[1], best[2]))

    os.makedirs("results", exist_ok=True)
    with open("results/sensitivity_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote results/sensitivity_table.csv (%d rows)" % len(rows))


if __name__ == "__main__":
    main()
