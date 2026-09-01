"""
estimator_autopsy.py - Diagnostics for the per-cube threshold ESTIMATION problem.

Task 12's H2/H3 story rests on the per-cube oracle headroom (diagnose_predictions'
percube_oracle). This script asks the two questions a reviewer will ask about it,
per dataset x model, on the TEST scenes:

  1. How much of the oracle headroom is REAL vs small-sample optimism?
     The per-cube oracle picks tau_i with knowledge of that cube's own labels, so
     with few points its F1 is optimistically biased (a max over noise). We split
     every cube's points 50/50 (seed 42): an oracle fitted on half A and applied
     to half B ("crossfit") keeps the oracle's information type (labels) but pays
     honest estimation variance. Reported on the B half, pooled over test scenes:
       oracleB_f1     in-sample oracle on B (the optimistic quantity, B-half analogue
                      of the reported percube_oracle)
       crossfit_f1    A-fitted per-cube taus applied to B (the honest quantity)
       globalB_f1     the DEPLOYED global f1_optimal tau applied to B (baseline)
       optimism_gap   oracleB - crossfit          (illusory part of the headroom)
       honest/reported headroom  crossfit-globalB vs oracleB-globalB
     Convention parity with diagnose_predictions: a cube whose fitting half has no
     positive points predicts none on the evaluation half.

  2. Is the oracle tau PREDICTABLE from label-free per-cube statistics at all?
     Pool test cubes (>=1 change point, >=1 no-change point, >=1 predicted-no-change
     point); regress the full-cube oracle tau on per-cube no-change features
     (count, mean, std, median, MAD, q90, q99, no-change fraction, cube size).
     5-fold CV R^2 for a linear model (the two_moment family lives here) and a
     random forest (an "any reasonable function of these features" bound), plus the
     Spearman correlation between the DEPLOYED two_moment taus and the oracle taus.

Outputs: results/estimator_autopsy.csv (upserted by dataset) + a printed table.
Torch-free, local CPU:  python src/estimator_autopsy.py --config configs/urb3dcd_v2_ld.yaml
"""

import os
import glob
import json
import csv
import argparse
import numpy as np
import yaml
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score

from calibrate import TAU_GRID, MIN_NC
from calibrate_v2 import OFFSET, ALPHA_GRID, load_split_scores, precompute_scene, cube_quantile

# ALPHA_GRID indices of the two tail quantiles used as regression features.
AI_90 = int(np.argmin(np.abs(ALPHA_GRID - 0.90)))
AI_99 = int(np.argmin(np.abs(ALPHA_GRID - 0.99)))

# The two_moment family sees only the no-change side of the score distribution.
# The change-side features below test whether the oracle tau needs BOTH sides
# (the oracle threshold sits in the gap between the two clusters).
FEATURE_NAMES = ["log_n_nc", "frac_nc", "mean_nc", "std_nc", "med_nc", "mad_nc",
                 "q90_nc", "q99_nc", "log_n_total",
                 "log_n_ch", "q05_ch", "q25_ch", "q50_ch"]


def change_side_stats(scores, inv, K):
    """Per-cube lower quantiles of the PREDICTED-change subset (score > 0.5, raw
    model call, no labels): the label-free view of where the change cluster starts."""
    sel = scores > 0.5
    vals = scores[sel]
    sub_inv = inv[sel]
    order = np.argsort(OFFSET * sub_inv + vals, kind="stable")   # sort by (cube, score)
    svals = vals[order]
    n = np.bincount(sub_inv, minlength=K).astype(np.int64)
    cum = np.concatenate([[0], np.cumsum(n)[:-1]])
    quants = [cube_quantile(svals, cum, n, a) for a in (0.05, 0.25, 0.50)]
    return n, quants


def class_keys(scores, inv, K, sel):
    """Sorted per-cube key structure for one point subset (one label class, possibly
    one half). Lets 'count of points above a per-cube tau' be two searchsorted calls."""
    key = np.sort(OFFSET * inv[sel] + scores[sel])
    n = np.bincount(inv[sel], minlength=K).astype(np.int64)
    cum = np.concatenate([[0], np.cumsum(n)[:-1]])       # cube start offsets in key
    return {"key": key, "n": n, "cum": cum}


def count_above(ck, tau_rows):
    """Per-cube count of subset points with score > tau, batched over candidate rows."""
    B, K = tau_rows.shape
    query = OFFSET * np.arange(K)[None, :] + tau_rows
    le = np.searchsorted(ck["key"], np.ascontiguousarray(query).ravel(),
                         side="right").reshape(B, K) - ck["cum"]
    return ck["n"][None, :] - le


def best_tau_per_cube(pos, neg, K):
    """Per-cube F1-optimal tau over TAU_GRID using labels (the oracle move).
    Returns (tau_best, valid) with valid = the cube has >=1 positive point."""
    tau_rows = np.broadcast_to(TAU_GRID[:, None], (len(TAU_GRID), K))
    tp = count_above(pos, tau_rows).astype(np.float64)
    fp = count_above(neg, tau_rows).astype(np.float64)
    fn = pos["n"][None, :] - tp
    precision = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1.0), 0.0)
    recall = np.where(tp + fn > 0, tp / np.maximum(tp + fn, 1.0), 0.0)
    both = precision + recall
    f1 = np.where(both > 0, 2 * precision * recall / np.maximum(both, 1e-300), 0.0)
    best_idx = np.argmax(f1, axis=0)                     # first max = lowest tau, as in diagnose
    return TAU_GRID[best_idx], pos["n"] > 0


def pooled_counts(pos, neg, tau_vec, valid):
    """(tp, fp, fn) pooled over cubes, where invalid cubes predict none (their
    positives count as fn, their negatives contribute no fp)."""
    tp = np.where(valid, count_above(pos, tau_vec[None, :])[0], 0)
    fp = np.where(valid, count_above(neg, tau_vec[None, :])[0], 0)
    return int(tp.sum()), int(fp.sum()), int(pos["n"].sum() - tp.sum())


def f1_from_counts(tp, fp, fn):
    """Change-class F1 from pooled counts."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0


def load_fitted(dataset, model):
    """Deployed global tau and two_moment (c, lambda, fallback) from the frozen Task 7 JSONs."""
    with open(os.path.join("calibration", dataset, model, "f1_optimal.json")) as f:
        global_tau = float(json.load(f)["fitted_params"]["tau"])
    with open(os.path.join("calibration", dataset, model, "two_moment.json")) as f:
        p = json.load(f)["fitted_params"]
    return global_tau, float(p["c"]), float(p["lambda"]), float(p["fallback_tau"])


def autopsy_model(test, dataset, model):
    """All autopsy quantities for one model's test scenes."""
    global_tau, tm_c, tm_lam, tm_fb = load_fitted(dataset, model)
    rng = np.random.default_rng(42)                      # one stream; scene order is sorted, so reproducible

    # Pooled split-half counts, accumulated across scenes.
    acc = {name: [0, 0, 0] for name in ["oracleB", "crossfit", "globalB", "oracle_full"]}
    n_cubes_total, n_pos_cubes = 0, 0
    feats, oracle_taus, tm_taus = [], [], []

    for rec in test:
        pre = precompute_scene(rec)                      # per-cube nc stats reused as features
        scores, labels, inv, K = rec["scores"], rec["labels"], pre["inv"], pre["K"]
        is_pos = labels == 1

        # Full-data oracle (should reproduce the discrimination csv's percube_oracle_pooled).
        pos_full = class_keys(scores, inv, K, is_pos)
        neg_full = class_keys(scores, inv, K, ~is_pos)
        tau_full, valid_full = best_tau_per_cube(pos_full, neg_full, K)
        tp, fp, fn = pooled_counts(pos_full, neg_full, tau_full, valid_full)
        acc["oracle_full"] = [a + b for a, b in zip(acc["oracle_full"], (tp, fp, fn))]

        # Split every point 50/50 into halves A (fit) and B (evaluate).
        in_a = rng.random(len(scores)) < 0.5
        pos_a = class_keys(scores, inv, K, is_pos & in_a)
        neg_a = class_keys(scores, inv, K, (~is_pos) & in_a)
        pos_b = class_keys(scores, inv, K, is_pos & ~in_a)
        neg_b = class_keys(scores, inv, K, (~is_pos) & ~in_a)

        tau_b, valid_b = best_tau_per_cube(pos_b, neg_b, K)          # in-sample oracle on B
        tau_a, valid_a = best_tau_per_cube(pos_a, neg_a, K)          # fitted on A, applied to B
        for name, (tau_vec, valid) in [("oracleB", (tau_b, valid_b)),
                                       ("crossfit", (tau_a, valid_a)),
                                       ("globalB", (np.full(K, global_tau), np.ones(K, bool)))]:
            tp, fp, fn = pooled_counts(pos_b, neg_b, tau_vec, valid)
            acc[name] = [a + b for a, b in zip(acc[name], (tp, fp, fn))]

        n_cubes_total += K
        n_pos_cubes += int(valid_full.sum())

        # Regression rows: cubes with both classes present and a nonempty nc subset.
        n_total = pos_full["n"] + neg_full["n"]
        n_ch, (q05_ch, q25_ch, q50_ch) = change_side_stats(scores, inv, K)
        use = valid_full & (neg_full["n"] > 0) & (pre["n_nc"] > 0)
        if use.any():
            f = np.column_stack([
                np.log1p(pre["n_nc"][use]),               # size of the nc population
                pre["n_nc"][use] / n_total[use],          # predicted no-change fraction
                pre["mean"][use], pre["std"][use],        # the two_moment ingredients
                pre["med"][use], pre["mad"][use],         # their robust counterparts
                pre["q_by_alpha"][AI_90][use], pre["q_by_alpha"][AI_99][use],
                np.log1p(n_total[use]),                   # cube population size
                np.log1p(n_ch[use]),                      # size of the predicted-change population
                q05_ch[use], q25_ch[use], q50_ch[use],    # where the change cluster starts
            ])
            feats.append(f)
            oracle_taus.append(tau_full[use])
            # The DEPLOYED two_moment tau for the same cubes (frozen c, lambda, fallback).
            tm = np.clip(tm_c + pre["mean"] + tm_lam * pre["std"], 0.0, 1.0)
            tm = np.where(pre["n_nc"] < MIN_NC, tm_fb, tm)
            tm_taus.append(tm[use])

    # Pooled F1s and the headroom decomposition on the B half.
    f1 = {name: f1_from_counts(*counts) for name, counts in acc.items()}
    reported = f1["oracleB"] - f1["globalB"]
    honest = f1["crossfit"] - f1["globalB"]
    row = {"dataset": dataset, "model": model,
           "n_test_cubes": n_cubes_total, "n_pos_cubes": n_pos_cubes,
           "deployed_tau": round(global_tau, 4),
           "oracle_full_f1": round(f1["oracle_full"], 4),
           "globalB_f1": round(f1["globalB"], 4),
           "oracleB_f1": round(f1["oracleB"], 4),
           "crossfit_f1": round(f1["crossfit"], 4),
           "optimism_gap": round(f1["oracleB"] - f1["crossfit"], 4),
           "reported_headroom_B": round(reported, 4),
           "honest_headroom_B": round(honest, 4),
           "honest_frac": round(honest / reported, 4) if reported > 1e-9 else ""}

    # Predictability of the oracle tau from label-free cube statistics.
    if feats and sum(len(x) for x in oracle_taus) >= 30:
        X = np.concatenate(feats)
        y = np.concatenate(oracle_taus)
        tm = np.concatenate(tm_taus)
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        X_nc = X[:, :9]                                   # the first 9 features are nc-side only

        def cv_r2(model, Xm):
            return round(float(np.mean(cross_val_score(model, Xm, y, cv=cv, scoring="r2"))), 4)

        rf = lambda: RandomForestRegressor(n_estimators=200, min_samples_leaf=5,
                                           random_state=42, n_jobs=-1)
        rho = spearmanr(tm, y).correlation                # deployed two_moment tau vs oracle tau
        row.update({"n_reg_cubes": len(y),
                    "linreg_cv_r2_nconly": cv_r2(LinearRegression(), X_nc),
                    "rf_cv_r2_nconly": cv_r2(rf(), X_nc),
                    "linreg_cv_r2_bothsides": cv_r2(LinearRegression(), X),
                    "rf_cv_r2_bothsides": cv_r2(rf(), X),
                    "spearman_tm_oracle": round(float(rho), 4) if np.isfinite(rho) else ""})
    else:
        row.update({"n_reg_cubes": sum(len(x) for x in oracle_taus),
                    "linreg_cv_r2_nconly": "", "rf_cv_r2_nconly": "",
                    "linreg_cv_r2_bothsides": "", "rf_cv_r2_bothsides": "",
                    "spearman_tm_oracle": ""})
    return row


def main():
    parser = argparse.ArgumentParser(description="Split-half oracle + oracle-tau predictability autopsy.")
    parser.add_argument("--config", required=True, help="dataset YAML in configs/")
    parser.add_argument("--models", nargs="*", default=None)
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

    fields = ["dataset", "model", "n_test_cubes", "n_pos_cubes", "deployed_tau",
              "oracle_full_f1", "globalB_f1", "oracleB_f1", "crossfit_f1", "optimism_gap",
              "reported_headroom_B", "honest_headroom_B", "honest_frac",
              "n_reg_cubes", "linreg_cv_r2_nconly", "rf_cv_r2_nconly",
              "linreg_cv_r2_bothsides", "rf_cv_r2_bothsides", "spearman_tm_oracle"]
    rows = []
    for model in models:
        _, test = load_split_scores(os.path.join(pred_root, model), split_of)
        row = autopsy_model(test, dataset, model)
        rows.append(row)
        print("%-18s oracle_full %.4f | B-half: global %.4f  crossfit %.4f  oracle %.4f "
              "| optimism %.4f  honest/reported %s | R2 nc-only lin %s rf %s  both lin %s rf %s  rho(tm,oracle) %s" % (
                  model, row["oracle_full_f1"], row["globalB_f1"], row["crossfit_f1"],
                  row["oracleB_f1"], row["optimism_gap"], str(row["honest_frac"]),
                  str(row["linreg_cv_r2_nconly"]), str(row["rf_cv_r2_nconly"]),
                  str(row["linreg_cv_r2_bothsides"]), str(row["rf_cv_r2_bothsides"]),
                  str(row["spearman_tm_oracle"])))

    # UPSERT by dataset, same discipline as the other results tables.
    out_path = os.path.join("results", "estimator_autopsy.csv")
    kept = []
    if os.path.exists(out_path):
        with open(out_path, newline="") as f:
            kept = [r for r in csv.DictReader(f) if r.get("dataset") != dataset]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in kept + rows:
            w.writerow(r)
    print("\nupserted %d %s rows into results/estimator_autopsy.csv (kept %d other rows)"
          % (len(rows), dataset, len(kept)))


if __name__ == "__main__":
    main()
