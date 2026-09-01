"""
calibrate_v2.py - Improved per-cube calibration methods (post-Task-12 method study).

Task 12 established that a real per-cube ORACLE headroom exists over a tuned global
threshold but the two-moment estimator realises almost none of it (H3). This script
tests four candidate estimators that each target one diagnosed failure mechanism,
on the SAME frozen predictions, with the SAME protocol as Task 7 calibrate.py
(fit on val by mean per-scene change-F1, apply to test, report val+test).

  quantile           tau_i = clip[0,1]( c + q_alpha(nochange_i) ). The per-cube
                     empirical quantile replaces the Gaussian mean+lambda*std
                     surrogate (scores are bounded/skewed, so moments estimate the
                     upper tail poorly). Conformal-style rank ceil(alpha*(n+1)).
                     TWO scalars (c, alpha) -- same parameter budget as two_moment.
                     NOTE the no-change subset is score<=0.5, so its scores are
                     truncated at 0.5 and a pure quantile can never exceed 0.5;
                     the global shift c compensates, exactly as it does for the
                     truncated mean in two_moment. Hard MIN_NC fallback kept.
  quantile_shrink    per-cube quantile shrunk toward the SCENE-pooled no-change
                     quantile with weight w_i = n_i/(n_i+k) (empirical-Bayes style).
                     THREE scalars (c, alpha, k). The hard MIN_NC fallback is
                     REPLACED by this soft fallback: as n_i -> 0 the threshold
                     degrades gracefully to the scene-level quantile + c, which is
                     the diagnosed IndoorCD failure (tiny no-change populations).
  two_moment_shrink  the original two-moment statistics shrunk the same way:
                     tau_i = clip( c + m_i' + lambda*s_i' ), m' and s' blended with
                     the scene-pooled no-change mean/std. THREE scalars (c, lambda, k).
                     No hard fallback. k=0 recovers plain two_moment (minus fallback).
  two_moment_robust  tau_i = clip( c + median_i + lambda*MAD_i ). Median/MAD resist
                     the contamination of the prediction-defined no-change subset by
                     missed changes (false negatives sit in its upper tail). MAD is
                     scaled by 1.4826 so lambda is comparable to the std version.
                     TWO scalars (c, lambda). Hard MIN_NC fallback kept.

Outputs (the frozen Task 7 artefacts are NEVER touched):
  calibration_v2/<dataset>/<model>/<method>.json    same schema as calibration/
  results/improved_methods_table.csv                same columns as main_table.csv, upserted
  results/improved_params_<dataset>.csv             fitted scalars per model x method

Torch-free, local CPU:  python src/calibrate_v2.py --config configs/urb3dcd_v2_ld.yaml
"""

import os
import glob
import json
import csv
import argparse
import numpy as np
import yaml

# Reuse the frozen Task 7 machinery so every convention (F1, objective, grids,
# fallback discipline, JSON schema) is identical by construction.
from calibrate import (TAU_GRID, MIN_NC, C_GRID, LAM_GRID, r, precision_recall_f1,
                       assemble, sweep_global_tau, cube_stats, load_diag)

# Candidate quantile levels for the conformal methods. 1.0 means the per-cube max.
ALPHA_GRID = np.array([0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90,
                       0.925, 0.95, 0.97, 0.99, 0.995, 0.999, 1.0])

# Candidate shrinkage strengths, in units of no-change points: w = n/(n+k).
# k=0 is pure per-cube, k=4096 halves the weight of even the largest cubes.
K_GRID = np.array([0.0, 4.0, 16.0, 64.0, 256.0, 1024.0, 4096.0])

# Standard consistency factor that makes MAD estimate the std under a Gaussian,
# so the lambda grid means the same thing for two_moment and two_moment_robust.
MAD_CONSIST = 1.4826

# Same key-offset trick as calibrate.precompute_fast_search: scores live in [0,1],
# so key = 2*cube + value keeps every cube in its own disjoint key band.
OFFSET = 2.0

METHODS_V2 = ["quantile", "quantile_shrink", "two_moment_shrink", "two_moment_robust"]


def load_split_scores(pred_dir, split_of):
    """Like calibrate.load_split but WITHOUT logits/coords (all methods here are
    score-only), which roughly halves memory on the big HKCD scenes."""
    val, test = [], []
    for path in sorted(glob.glob(os.path.join(pred_dir, "*.npz"))):
        d = np.load(path, allow_pickle=True)
        scene = str(d["scene_id"])
        split = split_of.get(scene, "?")
        if split not in ("val", "test"):
            continue                                     # skip without touching the arrays
        rec = {"scene": scene,
               "scores": d["scores"].astype(np.float64),  # f64 for stable stats, as in Task 7
               "labels": d["labels"].astype(np.int64),
               "cube_id": d["cube_id"]}
        (val if split == "val" else test).append(rec)
    return val, test


def rank_index(n, alpha):
    """Conformal empirical-quantile rank: index ceil(alpha*(n+1))-1, clipped to the
    sample. Works elementwise on integer arrays n."""
    idx = np.ceil(alpha * (n + 1)).astype(np.int64) - 1
    return np.clip(idx, 0, np.maximum(n - 1, 0))


def cube_quantile(sorted_vals, cum, n, alpha):
    """Per-cube empirical quantile from a per-cube-contiguous ascending value array.
    cum[i] is cube i's start offset, n[i] its count. Cubes with n=0 return 0.0."""
    if len(sorted_vals) == 0:
        return np.zeros(len(n))
    pos = np.minimum(cum + rank_index(n, alpha), len(sorted_vals) - 1)  # guard n=0 tails
    return np.where(n > 0, sorted_vals[pos], 0.0)


def pooled_quantile(sorted_vals, alpha):
    """Same rank convention on one pooled sorted array; 0.0 if it is empty."""
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    idx = min(max(int(np.ceil(alpha * (n + 1))) - 1, 0), n - 1)
    return float(sorted_vals[idx])


def shrink_weight(n, k):
    """Empirical-Bayes blend weight w = n/(n+k); at k=0 this is 1 for any n>0 and
    0 for n=0 (a cube with no no-change points carries no local information)."""
    n = n.astype(np.float64)
    if k <= 0.0:
        return (n > 0).astype(np.float64)
    return n / (n + k)


def precompute_scene(rec):
    """Everything every method needs for one scene, computed ONCE:
    fast-F1 label keys, per-cube no-change stats (count/mean/std/median/MAD and the
    sorted no-change scores for quantiles), and the scene-pooled no-change stats
    that the shrinkage methods blend toward."""
    scores = rec["scores"]
    labels = rec["labels"]
    stats = cube_stats(rec)                              # count/mean/std/inv/K, nochange = score<=0.5
    inv, K = stats["inv"], stats["K"]

    # --- fast-F1 structures (same layout as calibrate.precompute_fast_search) ---
    is_pos = labels == 1
    pos_key = np.sort(OFFSET * inv[is_pos] + scores[is_pos])
    neg_key = np.sort(OFFSET * inv[~is_pos] + scores[~is_pos])
    n_pos = np.bincount(inv[is_pos], minlength=K).astype(np.int64)
    n_neg = np.bincount(inv[~is_pos], minlength=K).astype(np.int64)
    cum_pos = np.concatenate([[0], np.cumsum(n_pos)[:-1]])   # cube start offsets in pos_key
    cum_neg = np.concatenate([[0], np.cumsum(n_neg)[:-1]])

    # --- per-cube no-change score structures (RAW model call, no labels) ---
    nc_mask = scores <= 0.5
    nc_inv = inv[nc_mask]
    nc_scores = scores[nc_mask]
    order = np.argsort(OFFSET * nc_inv + nc_scores, kind="stable")   # sort by (cube, score)
    nc_sorted = nc_scores[order]                         # per-cube-contiguous, ascending in cube
    n_nc = stats["count"].astype(np.int64)
    cum_nc = np.concatenate([[0], np.cumsum(n_nc)[:-1]])

    # Per-cube quantile matrix over the whole ALPHA_GRID (rows = alpha, cols = cube).
    q_by_alpha = np.stack([cube_quantile(nc_sorted, cum_nc, n_nc, a) for a in ALPHA_GRID])

    # Per-cube median and MAD. Absolute deviations are <= 0.5 (scores <= 0.5), so the
    # same OFFSET=2 key trick separates cubes for the second sort too.
    med = cube_quantile(nc_sorted, cum_nc, n_nc, 0.5)
    absdev = np.abs(nc_scores - med[nc_inv])
    ad_sorted = np.sort(OFFSET * nc_inv + absdev)        # per-cube-contiguous sorted deviations
    ad_sorted = ad_sorted - OFFSET * np.repeat(np.arange(K), n_nc)   # strip the cube offsets back off
    mad = MAD_CONSIST * cube_quantile(ad_sorted, cum_nc, n_nc, 0.5)

    # --- scene-pooled no-change stats (shrinkage targets). A scene where the model
    # calls EVERYTHING change (e.g. MS pointnet) has no no-change points at all;
    # the pooled stats degenerate to 0 and tau degenerates to the constant c. ---
    if len(nc_scores) > 0:
        g_mean = float(nc_scores.mean())
        g_std = float(nc_scores.std())
        pooled_sorted = np.sort(nc_scores)
        g_quant = np.array([pooled_quantile(pooled_sorted, a) for a in ALPHA_GRID])
        del pooled_sorted                                # only the extracted quantiles are kept
    else:
        g_mean, g_std = 0.0, 0.0
        g_quant = np.zeros(len(ALPHA_GRID))

    return {"K": K, "inv": inv, "n_nc": n_nc,
            "mean": stats["mean"], "std": stats["std"], "med": med, "mad": mad,
            "q_by_alpha": q_by_alpha, "g_mean": g_mean, "g_std": g_std, "g_quant": g_quant,
            "pos_key": pos_key, "neg_key": neg_key, "n_pos": n_pos, "n_neg": n_neg,
            "cum_pos": cum_pos, "cum_neg": cum_neg, "total_pos": int(n_pos.sum())}


def f1_batch(pre, tau_rows):
    """Change-class F1 for a BATCH of per-cube threshold vectors (one row = one
    candidate), via two searchsorted calls. Same strict 'score > tau' counting as
    calibrate.scene_f1_fast; this only accelerates SELECTION -- reported numbers
    always come from the per-point predictions in assemble()."""
    B, K = tau_rows.shape
    query = OFFSET * np.arange(K)[None, :] + tau_rows    # key per (candidate, cube)
    pos_le = np.searchsorted(pre["pos_key"], query.ravel(), side="right").reshape(B, K) - pre["cum_pos"]
    neg_le = np.searchsorted(pre["neg_key"], query.ravel(), side="right").reshape(B, K) - pre["cum_neg"]
    tp = (pre["n_pos"] - pos_le).sum(axis=1).astype(np.float64)   # points above tau, per class
    fp = (pre["n_neg"] - neg_le).sum(axis=1).astype(np.float64)
    fn = pre["total_pos"] - tp
    precision = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1.0), 0.0)
    recall = np.where(tp + fn > 0, tp / np.maximum(tp + fn, 1.0), 0.0)
    both = precision + recall
    return np.where(both > 0, 2 * precision * recall / np.maximum(both, 1e-300), 0.0)


# ---------------------------------------------------------------------------
# Per-cube tau builders. Each is THE single definition of its method's threshold,
# used both during selection (via f1_batch) and for the final reported predictions.
# ---------------------------------------------------------------------------

def tau_quantile(pre, c, alpha_idx, fb_tau):
    """Conformal quantile + global shift, hard MIN_NC fallback like two_moment."""
    tau = np.clip(c + pre["q_by_alpha"][alpha_idx], 0.0, 1.0)
    return np.where(pre["n_nc"] < MIN_NC, fb_tau, tau)


def tau_quantile_shrink(pre, c, alpha_idx, k):
    """Quantile blended toward the scene-pooled quantile; no hard fallback."""
    w = shrink_weight(pre["n_nc"], k)
    blended = w * pre["q_by_alpha"][alpha_idx] + (1.0 - w) * pre["g_quant"][alpha_idx]
    return np.clip(c + blended, 0.0, 1.0)


def tau_two_moment_shrink(pre, c, lam, k):
    """Two-moment with mean/std blended toward the scene-pooled moments."""
    w = shrink_weight(pre["n_nc"], k)
    mean_sh = w * pre["mean"] + (1.0 - w) * pre["g_mean"]
    std_sh = w * pre["std"] + (1.0 - w) * pre["g_std"]
    return np.clip(c + mean_sh + lam * std_sh, 0.0, 1.0)


def tau_two_moment_robust(pre, c, lam, fb_tau):
    """Median/MAD version of two_moment, hard MIN_NC fallback kept."""
    tau = np.clip(c + pre["med"] + lam * pre["mad"], 0.0, 1.0)
    return np.where(pre["n_nc"] < MIN_NC, fb_tau, tau)


# ---------------------------------------------------------------------------
# Fitting: each method sweeps its grid on val, batching the c dimension so one
# f1_batch call scores all 61 c values at once.
# ---------------------------------------------------------------------------

def fit_quantile(val_pre, fb_tau):
    """Grid over (alpha, c); returns (c, alpha_idx, val_f1)."""
    best = (-1.0, 0.0, 0)
    for ai in range(len(ALPHA_GRID)):
        f1_sum = np.zeros(len(C_GRID))
        for pre in val_pre:
            base = pre["q_by_alpha"][ai]                 # per-cube quantile for this alpha
            rows = np.clip(C_GRID[:, None] + base[None, :], 0.0, 1.0)
            rows = np.where(pre["n_nc"][None, :] < MIN_NC, fb_tau, rows)
            f1_sum += f1_batch(pre, rows)
        mean_f1 = f1_sum / len(val_pre)
        ci = int(np.argmax(mean_f1))
        if mean_f1[ci] > best[0]:
            best = (float(mean_f1[ci]), float(C_GRID[ci]), ai)
    return best[1], best[2], best[0]


def fit_quantile_shrink(val_pre):
    """Grid over (alpha, k, c); returns (c, alpha_idx, k, val_f1)."""
    best = (-1.0, 0.0, 0, 0.0)
    for ai in range(len(ALPHA_GRID)):
        for k in K_GRID:
            f1_sum = np.zeros(len(C_GRID))
            for pre in val_pre:
                w = shrink_weight(pre["n_nc"], k)
                base = w * pre["q_by_alpha"][ai] + (1.0 - w) * pre["g_quant"][ai]
                rows = np.clip(C_GRID[:, None] + base[None, :], 0.0, 1.0)
                f1_sum += f1_batch(pre, rows)
            mean_f1 = f1_sum / len(val_pre)
            ci = int(np.argmax(mean_f1))
            if mean_f1[ci] > best[0]:
                best = (float(mean_f1[ci]), float(C_GRID[ci]), ai, float(k))
    return best[1], best[2], best[3], best[0]


def fit_two_moment_shrink(val_pre):
    """Grid over (k, lambda, c); returns (c, lambda, k, val_f1)."""
    best = (-1.0, 0.0, 0.0, 0.0)
    for k in K_GRID:
        # Blend the moments once per (scene, k); the lambda/c sweep reuses them.
        blended = []
        for pre in val_pre:
            w = shrink_weight(pre["n_nc"], k)
            blended.append((w * pre["mean"] + (1.0 - w) * pre["g_mean"],
                            w * pre["std"] + (1.0 - w) * pre["g_std"]))
        for lam in LAM_GRID:
            f1_sum = np.zeros(len(C_GRID))
            for pre, (m_sh, s_sh) in zip(val_pre, blended):
                rows = np.clip(C_GRID[:, None] + (m_sh + lam * s_sh)[None, :], 0.0, 1.0)
                f1_sum += f1_batch(pre, rows)
            mean_f1 = f1_sum / len(val_pre)
            ci = int(np.argmax(mean_f1))
            if mean_f1[ci] > best[0]:
                best = (float(mean_f1[ci]), float(C_GRID[ci]), float(lam), float(k))
    return best[1], best[2], best[3], best[0]


def fit_two_moment_robust(val_pre, fb_tau):
    """Grid over (lambda, c) on median/MAD; returns (c, lambda, val_f1)."""
    best = (-1.0, 0.0, 0.0)
    for lam in LAM_GRID:
        f1_sum = np.zeros(len(C_GRID))
        for pre in val_pre:
            base = pre["med"] + lam * pre["mad"]
            rows = np.clip(C_GRID[:, None] + base[None, :], 0.0, 1.0)
            rows = np.where(pre["n_nc"][None, :] < MIN_NC, fb_tau, rows)
            f1_sum += f1_batch(pre, rows)
        mean_f1 = f1_sum / len(val_pre)
        ci = int(np.argmax(mean_f1))
        if mean_f1[ci] > best[0]:
            best = (float(mean_f1[ci]), float(C_GRID[ci]), float(lam))
    return best[1], best[2], best[0]


def run_method(method, model, dataset, val, test, pre_by_scene, fb_tau):
    """Fit one method on val, assemble the full JSON result via the Task 7 assemble().
    Returns (result_dict, fitted_params_row)."""
    val_pre = [pre_by_scene[rec["scene"]] for rec in val]

    # Fit and freeze this method's per-cube tau function (single source of truth).
    if method == "quantile":
        c, ai, vf1 = fit_quantile(val_pre, fb_tau)
        tau_fn = lambda pre: tau_quantile(pre, c, ai, fb_tau)
        fitted = {"c": r(c), "alpha": float(ALPHA_GRID[ai]), "min_nc": MIN_NC,
                  "fallback_tau": r(fb_tau), "val_fit_mean_f1": r(vf1)}
    elif method == "quantile_shrink":
        c, ai, k, vf1 = fit_quantile_shrink(val_pre)
        tau_fn = lambda pre: tau_quantile_shrink(pre, c, ai, k)
        fitted = {"c": r(c), "alpha": float(ALPHA_GRID[ai]), "k": k,
                  "fallback": "soft (shrink to scene quantile)", "val_fit_mean_f1": r(vf1)}
    elif method == "two_moment_shrink":
        c, lam, k, vf1 = fit_two_moment_shrink(val_pre)
        tau_fn = lambda pre: tau_two_moment_shrink(pre, c, lam, k)
        fitted = {"c": r(c), "lambda": r(lam), "k": k,
                  "fallback": "soft (shrink to scene moments)", "val_fit_mean_f1": r(vf1)}
    elif method == "two_moment_robust":
        c, lam, vf1 = fit_two_moment_robust(val_pre, fb_tau)
        tau_fn = lambda pre: tau_two_moment_robust(pre, c, lam, fb_tau)
        fitted = {"c": r(c), "lambda": r(lam), "mad_consistency": MAD_CONSIST,
                  "min_nc": MIN_NC, "fallback_tau": r(fb_tau), "val_fit_mean_f1": r(vf1)}
    else:
        raise ValueError(method)
    fitted["nochange_subset"] = "score<=0.5 (raw prediction, no labels)"

    # Reported numbers come from plain per-point comparisons, never the fast path.
    predict_fn = lambda rec: rec["scores"] > tau_fn(pre_by_scene[rec["scene"]])[pre_by_scene[rec["scene"]]["inv"]]
    tau_of_scene = lambda rec: [r(v) for v in tau_fn(pre_by_scene[rec["scene"]])]
    result = assemble(model, method, dataset, val, test, predict_fn, tau_of_scene, fitted)

    # Record cube counts plus the fallback pressure (hard count or mean soft weight).
    for rec in (val + test):
        pre = pre_by_scene[rec["scene"]]
        result["per_scene"][rec["scene"]]["n_cubes"] = int(pre["K"])
        if method in ("quantile", "two_moment_robust"):
            result["per_scene"][rec["scene"]]["n_fallback"] = int(np.sum(pre["n_nc"] < MIN_NC))
        else:
            k = fitted["k"]
            result["per_scene"][rec["scene"]]["mean_shrink_w"] = r(float(np.mean(shrink_weight(pre["n_nc"], k))))
    return result, fitted


def main():
    parser = argparse.ArgumentParser(description="Run the improved per-cube calibration methods.")
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
    ceiling, _ = load_diag(dataset)

    # Reference numbers from the frozen main table, for the headline comparison.
    reference = {}
    main_path = os.path.join("results", "main_table.csv")
    if os.path.exists(main_path):
        with open(main_path, newline="") as f:
            for row in csv.DictReader(f):
                if row["dataset"] == dataset and row["method"] in ("f1_optimal", "two_moment"):
                    reference[(row["model"], row["method"])] = float(row["test_mean_f1"])

    table_rows = []
    param_rows = []
    for model in models:
        val, test = load_split_scores(os.path.join(pred_root, model), split_of)
        print("\n== %s ==  (val %d scenes, test %d scenes)" % (model, len(val), len(test)))

        # Same fallback tau as Task 7: the global f1_optimal threshold refit on val.
        fb_tau, _ = sweep_global_tau(val, lambda rec: rec["scores"])

        # One precompute per scene, shared by all four methods.
        pre_by_scene = {rec["scene"]: precompute_scene(rec) for rec in (val + test)}

        results = {}
        for method in METHODS_V2:
            results[method], fitted = run_method(method, model, dataset, val, test, pre_by_scene, fb_tau)
            param_rows.append({"dataset": dataset, "model": model, "method": method,
                               "c": fitted.get("c", ""), "lambda": fitted.get("lambda", ""),
                               "alpha": fitted.get("alpha", ""), "k": fitted.get("k", ""),
                               "fallback_tau": fitted.get("fallback_tau", ""),
                               "val_fit_mean_f1": fitted["val_fit_mean_f1"]})

        out_dir = os.path.join("calibration_v2", dataset, model)
        os.makedirs(out_dir, exist_ok=True)
        for method in METHODS_V2:
            with open(os.path.join(out_dir, method + ".json"), "w") as f:
                json.dump(results[method], f, indent=2)

        # Per-model summary against the frozen references.
        ref_fo = reference.get((model, "f1_optimal"), float("nan"))
        ref_tm = reference.get((model, "two_moment"), float("nan"))
        cap = ceiling.get(model, float("nan"))
        print("  reference: f1_optimal %.4f  two_moment %.4f  oracle ceiling %.4f" % (ref_fo, ref_tm, cap))
        print("  %-20s %8s %8s %9s %9s" % ("method", "valF1", "testF1", "vs_f1opt", "vs_twomom"))
        for method in METHODS_V2:
            s = results[method]["summary"]
            table_rows.append({
                "dataset": dataset, "model": model, "method": method,
                "val_mean_p": s["val_mean_p"], "val_mean_r": s["val_mean_r"], "val_mean_f1": s["val_mean_f1"],
                "test_mean_p": s["test_mean_p"], "test_mean_r": s["test_mean_r"], "test_mean_f1": s["test_mean_f1"],
                "val_pool_p": s["val_pooled"]["precision"], "val_pool_r": s["val_pooled"]["recall"],
                "val_pool_f1": s["val_pooled"]["f1"],
                "test_pool_p": s["test_pooled"]["precision"], "test_pool_r": s["test_pooled"]["recall"],
                "test_pool_f1": s["test_pooled"]["f1"],
            })
            print("  %-20s %8.4f %8.4f %+9.4f %+9.4f" % (
                method, s["val_mean_f1"], s["test_mean_f1"],
                s["test_mean_f1"] - ref_fo, s["test_mean_f1"] - ref_tm))

    # UPSERT into the improved-methods table (same discipline as main_table.csv).
    os.makedirs("results", exist_ok=True)
    table_fields = ["dataset", "model", "method", "val_mean_p", "val_mean_r", "val_mean_f1",
                    "test_mean_p", "test_mean_r", "test_mean_f1", "val_pool_p", "val_pool_r",
                    "val_pool_f1", "test_pool_p", "test_pool_r", "test_pool_f1"]
    out_path = os.path.join("results", "improved_methods_table.csv")
    kept = []
    if os.path.exists(out_path):
        with open(out_path, newline="") as f:
            kept = [row for row in csv.DictReader(f) if row.get("dataset") != dataset]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=table_fields, extrasaction="ignore")
        w.writeheader()
        for row in kept + table_rows:
            w.writerow(row)

    with open(os.path.join("results", "improved_params_" + dataset + ".csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "model", "method", "c", "lambda",
                                          "alpha", "k", "fallback_tau", "val_fit_mean_f1"])
        w.writeheader()
        for row in param_rows:
            w.writerow(row)

    assert len(table_rows) == len(models) * len(METHODS_V2), "missing model x method rows"
    print("\nupserted %d %s rows into results/improved_methods_table.csv (kept %d other rows)"
          % (len(table_rows), dataset, len(kept)))
    print("wrote results/improved_params_%s.csv and calibration_v2/%s/<model>/<method>.json" % (dataset, dataset))


if __name__ == "__main__":
    main()
