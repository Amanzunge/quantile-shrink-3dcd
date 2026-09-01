"""
transfer_v2.py - Task 10 re-run for BOTH per-cube methods (two_moment and the
winning calibrate_v2 method, quantile_shrink).

Protocol is IDENTICAL to Task 10 transfer.py: take the scalars fit on a SOURCE
dataset's val (two_moment: c, lambda; quantile_shrink: c, alpha, k), apply them
UNCHANGED to the TARGET dataset's TEST predictions (the per-cube statistics are
always computed from the TARGET's own test-time predictions), and compare against
the target-refit version and the target-tuned f1_optimal.

This SUPERSEDES the old results/transfer_table.csv (which held only two_moment):
the two_moment rows are recomputed here by the same code path, so the old
information is preserved inside the new table under method=two_moment.

  python src/transfer_v2.py

Writes results/transfer_table.csv with a 'method' column (32 rows = 16 pairs x 2).
"""

import os
import csv
import argparse
import numpy as np

# Frozen Task 7 / Task 10 machinery.
from calibrate import cube_stats, two_moment_pred, load_diag
from diagnose_predictions import f1_from_pred
from transfer import CONFIG_OF, FAILED, r, split_of_dataset, load_two_moment_params, load_main_table_refs
# calibrate_v2 machinery for the quantile_shrink arm.
from calibrate_v2 import ALPHA_GRID, load_split_scores, precompute_scene, tau_quantile_shrink


def load_qs_params(dataset):
    """results/improved_params_<dataset>.csv -> {model: {'c','alpha','k'}} for quantile_shrink."""
    path = os.path.join("results", "improved_params_" + dataset + ".csv")
    out = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row["method"] == "quantile_shrink":
                out[row["model"]] = {"c": float(row["c"]), "alpha": float(row["alpha"]),
                                     "k": float(row["k"])}
    return out


def load_v2_refs(dataset):
    """results/improved_methods_table.csv -> {model: quantile_shrink test_mean_f1} (refit cross-check)."""
    out = {}
    with open(os.path.join("results", "improved_methods_table.csv"), newline="") as f:
        for row in csv.DictReader(f):
            if row["dataset"] == dataset and row["method"] == "quantile_shrink":
                out[row["model"]] = float(row["test_mean_f1"])
    return out


def alpha_index(alpha):
    """Map a stored alpha value back to its ALPHA_GRID index."""
    return int(np.argmin(np.abs(ALPHA_GRID - alpha)))


def mean_test_f1_tm(test_recs, stats_by_scene, c, lam, fb_tau):
    """two_moment mean per-scene test F1 (same as transfer.py mean_test_f1)."""
    f1s = []
    for rec in test_recs:
        pred = two_moment_pred(stats_by_scene[rec["scene"]], rec["scores"], c, lam, fb_tau)
        f1s.append(f1_from_pred(pred, rec["labels"]))
    return float(np.mean(f1s))


def mean_test_f1_qs(test_recs, pre_by_scene, c, alpha, k):
    """quantile_shrink mean per-scene test F1 under given (c, alpha, k)."""
    ai = alpha_index(alpha)
    f1s = []
    for rec in test_recs:
        pre = pre_by_scene[rec["scene"]]
        tau = tau_quantile_shrink(pre, c, ai, k)          # per-cube tau from TARGET statistics
        f1s.append(f1_from_pred(rec["scores"] > tau[pre["inv"]], rec["labels"]))
    return float(np.mean(f1s))


def main():
    parser = argparse.ArgumentParser(description="Task 10 transfer for two_moment AND quantile_shrink.")
    parser.add_argument("--out", default=os.path.join("results", "transfer_table.csv"))
    args = parser.parse_args()

    # Same 16 experiment pairs as Task 10 transfer.py.
    specs = []
    ld_models = ["icp_euclidean", "siamese_pointnet", "siamese_pointnet2", "siamese_kpconv", "siamgcn"]
    for m in ld_models:
        specs.append(("urb3dcd_v2_ld", "hkcd", m, True))
    specs.append(("urb3dcd_v2_ms", "hkcd", "randla", True))
    for m in ld_models:
        specs.append(("urb3dcd_v2_ms", "hkcd", m, False))
    for m in ld_models:
        specs.append(("urb3dcd_v2_ld", "urb3dcd_v2_ms", m, False))

    rows = []
    test_cache = {}                                       # (target, model) -> (recs, stats, pre)
    refs_cache, v2refs_cache, oracle_cache = {}, {}, {}

    for source, target, model, is_primary in specs:
        tm_src_all = load_two_moment_params(source)
        tm_tgt_all = load_two_moment_params(target)
        qs_src_all = load_qs_params(source)
        qs_tgt_all = load_qs_params(target)
        if model not in tm_src_all or model not in tm_tgt_all:
            continue                                      # model absent on one end (randla on LD)

        # Target test scenes, loaded once per (target, model); both methods share them.
        if (target, model) not in test_cache:
            split_of = split_of_dataset(target)
            _val, test_recs = load_split_scores(os.path.join("predictions", target, model), split_of)
            stats_by_scene = {rec["scene"]: cube_stats(rec) for rec in test_recs}
            pre_by_scene = {rec["scene"]: precompute_scene(rec) for rec in test_recs}
            test_cache[(target, model)] = (test_recs, stats_by_scene, pre_by_scene)
        test_recs, stats_by_scene, pre_by_scene = test_cache[(target, model)]

        if target not in refs_cache:
            refs_cache[target] = load_main_table_refs(target)
            v2refs_cache[target] = load_v2_refs(target)
            oracle_cache[target] = load_diag(target)[0]
        refs = refs_cache[target]

        # Health status is a property of the (source, target, model) pair, not the method.
        if model in FAILED.get(source, set()):
            status = "na_source_failed"
        elif model in FAILED.get(target, set()):
            status = "na_target_failed"
        else:
            status = "valid"

        fixed_05_f1 = refs.get((model, "fixed_05"), float("nan"))
        f1_optimal_f1 = refs.get((model, "f1_optimal"), float("nan"))
        oracle_f1 = oracle_cache[target].get(model, float("nan"))
        headroom = oracle_f1 - f1_optimal_f1

        for method in ("two_moment", "quantile_shrink"):
            if method == "two_moment":
                sp, tp = tm_src_all[model], tm_tgt_all[model]
                # Fallback tau = TARGET's, so the arms differ ONLY in (c, lambda) (as in Task 10).
                transferred = mean_test_f1_tm(test_recs, stats_by_scene, sp["c"], sp["lambda"], tp["fallback_tau"])
                refit = mean_test_f1_tm(test_recs, stats_by_scene, tp["c"], tp["lambda"], tp["fallback_tau"])
                refit_ref = refs.get((model, "two_moment"))
                src_desc = {"src_c": r(sp["c"]), "src_lambda": r(sp["lambda"]), "src_alpha": "", "src_k": ""}
                tgt_desc = {"tgt_c": r(tp["c"]), "tgt_lambda": r(tp["lambda"]), "tgt_alpha": "", "tgt_k": ""}
            else:
                sp, tp = qs_src_all[model], qs_tgt_all[model]
                transferred = mean_test_f1_qs(test_recs, pre_by_scene, sp["c"], sp["alpha"], sp["k"])
                refit = mean_test_f1_qs(test_recs, pre_by_scene, tp["c"], tp["alpha"], tp["k"])
                refit_ref = v2refs_cache[target].get(model)
                src_desc = {"src_c": r(sp["c"]), "src_lambda": "", "src_alpha": sp["alpha"], "src_k": sp["k"]}
                tgt_desc = {"tgt_c": r(tp["c"]), "tgt_lambda": "", "tgt_alpha": tp["alpha"], "tgt_k": tp["k"]}
            if refit_ref is not None and abs(refit - refit_ref) > 5e-3:
                print("  WARN recomputed %s refit %.4f != table %.4f (%s/%s)" % (
                    method, refit, refit_ref, target, model))

            gap = refit - transferred
            tr_minus_f1opt = transferred - f1_optimal_f1
            row = {"method": method, "source": source, "target": target, "model": model,
                   "status": status, "is_primary": int(is_primary),
                   "transferred_f1": r(transferred), "refit_f1": r(refit), "gap": r(gap),
                   "fixed_05_f1": r(fixed_05_f1), "f1_optimal_f1": r(f1_optimal_f1),
                   "oracle_f1": r(oracle_f1), "transferred_minus_f1opt": r(tr_minus_f1opt),
                   "headroom": r(headroom),
                   "frac_headroom_kept": r(tr_minus_f1opt / headroom) if headroom > 1e-9 else ""}
            row.update(src_desc)
            row.update(tgt_desc)
            rows.append(row)

    fields = ["method", "source", "target", "model", "status", "is_primary",
              "src_c", "src_lambda", "src_alpha", "src_k",
              "tgt_c", "tgt_lambda", "tgt_alpha", "tgt_k",
              "transferred_f1", "refit_f1", "gap",
              "fixed_05_f1", "f1_optimal_f1", "oracle_f1",
              "transferred_minus_f1opt", "headroom", "frac_headroom_kept"]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print("wrote %s (%d rows)" % (args.out, len(rows)))

    # Verdict on the valid PRIMARY rows, per method.
    for method in ("two_moment", "quantile_shrink"):
        sel = [x for x in rows if x["method"] == method and x["is_primary"] == 1 and x["status"] == "valid"]
        if not sel:
            continue
        gaps = [x["gap"] for x in sel]
        deltas = [x["transferred_minus_f1opt"] for x in sel]
        print("\n== %s PRIMARY verdict (n=%d valid sim->real rows) ==" % (method, len(sel)))
        for x in sel:
            print("  %-14s -> %-6s %-18s transf %.4f  refit %.4f  gap %+.4f  vs f1opt %+.4f" % (
                x["source"].replace("urb3dcd_v2_", ""), x["target"], x["model"],
                x["transferred_f1"], x["refit_f1"], x["gap"], x["transferred_minus_f1opt"]))
        print("  mean gap (refit - transferred) = %+.4f | mean (transferred - f1opt) = %+.4f | >=f1opt on %d/%d"
              % (float(np.mean(gaps)), float(np.mean(deltas)),
                 int(np.sum(np.array(deltas) >= -1e-9)), len(deltas)))


if __name__ == "__main__":
    main()
