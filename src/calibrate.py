"""
calibrate.py - Task 7. Implement all six calibration methods (CLAUDE Section 3),
run them on the saved predictions for every model of one dataset, and write:
  - calibration/<dataset>/<model>/<method>.json   (fitted params, per-scene tau + P/R/F1)
  - results/main_table.csv                          (model x method, val/test F1)
  - results/two_moment_params_<dataset>.csv         (the (c, lambda) Task 10 transfers)

The decision is applied PER POINT. Methods differ ONLY in how the threshold tau is
set. The metric is F1 of the CHANGE class. Everything is fit on VAL and reported on
VAL and TEST. Torch-free (numpy + sklearn + scipy). Local CPU analysis (Section 12).

  python src/calibrate.py --config configs/urb3dcd_v2_ld.yaml

Method summary (Section 3, with the decision-threshold conventions confirmed for Task 7):
  fixed_05             tau = 0.5 on raw scores. No fitting.
  f1_optimal           one global tau swept on val to max mean per-scene val F1.
  temperature_scaling  scalar T fit on val by min NLL of the 2-class logits, then an
                       F1-optimal tau on the calibrated probs (CLAUDE wording). Being a
                       monotonic rescale it reduces to ONE global score threshold, so it
                       collapses onto f1_optimal up to grid resolution (reported honestly).
  platt                A, B fit by logistic regression of the change log-odds vs labels;
                       p = sigmoid(A*z + B); decide at the fixed 0.5 operating point.
  isotonic             monotonic score->label map fit on pooled val; decide at fixed 0.5.
  two_moment (OURS)    per-cube tau_i = clip[0,1]( c + mean_nochange_i + lambda*std_nochange_i ).
                       The no-change subset of a cube is its points with score <= 0.5 (RAW
                       model call, NO labels). (c, lambda) are two scalars fit ONCE per model
                       on val by 2D grid search maximizing mean per-scene val F1.
"""

import os
import glob
import json
import csv
import argparse
import numpy as np
import yaml
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

# Reuse the diagnostic's F1 helper so the table matches its conventions exactly.
from diagnose_predictions import f1_from_pred


# Same threshold grid the diagnostic sweeps, so f1_optimal's val F1 lines up with its best_global.
TAU_GRID = np.linspace(0.02, 0.98, 49)

# two_moment fallback: a cube with fewer than this many no-change points has an unstable
# std and uses the global f1_optimal tau instead (confirmed min_nc for Task 7).
MIN_NC = 10

# two_moment (c, lambda) search grid. Step is the confirmed 0.02 / 0.1. The grid was widened
# twice in response to boundary-pinned optima (the sanctioned remedy when an optimum lands on
# an edge; the fitted (c, lambda) are a reported artifact + Task 10 transfers them, so they
# must be genuine interior optima):
#   - Task 7 (LD): c upper 0.4 -> 0.8 after c pinned at 0.4 for three models.
#   - Task 8 (MS): lambda upper 4.0 -> 8.0 after MS ICP pinned lambda at 3.9 (no-change score
#     means are low ~0.25 while the needed threshold is high ~0.88, so the std term has to lift
#     tau a long way); c upper 0.8 -> 1.0 preemptively (tau clips at 1.0, so 1.0 is the useful
#     max) in case a confident deep model on MS needs a large flat offset.
# One uniform grid across datasets keeps the LD/MS/HKCD fits comparable; LD optima are deep
# interior (lambda ~0, c 0.1-0.62), so the wider grid reproduces the LD values exactly.
C_GRID = np.linspace(-0.2, 1.0, 61)
LAM_GRID = np.linspace(0.0, 8.0, 81)


def r(x, n=4):
    """Round to a plain python float so json can serialize it."""
    return round(float(x), n)


def precision_recall_f1(pred, labels):
    """P/R/F1 of the change class. Same tp/fp/fn convention as diagnose_predictions.f1_from_pred."""
    tp = int(np.sum(pred & (labels == 1)))
    fp = int(np.sum(pred & (labels == 0)))
    fn = int(np.sum((~pred) & (labels == 1)))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def load_split(pred_dir, split_of):
    """Load every scene npz for one model into val and test lists of dicts."""
    val, test = [], []
    for path in sorted(glob.glob(os.path.join(pred_dir, "*.npz"))):
        d = np.load(path, allow_pickle=True)
        scene = str(d["scene_id"])
        rec = {
            "scene": scene,
            "scores": d["scores"].astype(np.float64),   # f64 for stable fits and stats
            "logits": d["logits"].astype(np.float64),    # raw 2-class logits (temp/platt input)
            "labels": d["labels"].astype(np.int64),
            "cube_id": d["cube_id"],                     # per-scene 0..K-1, NOT unique across scenes
        }
        split = split_of.get(scene, "?")
        if split == "val":
            val.append(rec)
        elif split == "test":
            test.append(rec)
    return val, test


def mean_perscene_f1(scenes, predict_fn):
    """Mean over scenes of per-scene change-F1, the shared val objective for the fits."""
    f1s = [f1_from_pred(predict_fn(rec), rec["labels"]) for rec in scenes]
    return float(np.mean(f1s))


def sweep_global_tau(scenes, score_fn):
    """Pick the single tau over TAU_GRID that maximizes mean per-scene val F1.
    score_fn(rec) returns the per-point deciding values (raw scores or calibrated probs)."""
    best_tau, best_f1 = 0.5, -1.0
    for tau in TAU_GRID:
        f1 = mean_perscene_f1(scenes, lambda rec, t=tau: score_fn(rec) > t)
        if f1 > best_f1:
            best_f1, best_tau = f1, float(tau)
    return best_tau, best_f1


def assemble(model, method, dataset, val, test, predict_fn, tau_of_scene, fitted_params):
    """Run predict_fn on val and test, build the per-scene + summary JSON dict for one method."""
    per_scene = {}
    summary = {}
    for scenes, split in [(val, "val"), (test, "test")]:
        f1s, ps, rs = [], [], []
        all_pred, all_lab = [], []
        for rec in scenes:
            pred = predict_fn(rec)
            precision, recall, f1 = precision_recall_f1(pred, rec["labels"])
            per_scene[rec["scene"]] = {
                "split": split,
                "tau": tau_of_scene(rec),               # scalar for global methods, per-cube list for two_moment
                "precision": r(precision), "recall": r(recall), "f1": r(f1),
            }
            f1s.append(f1); ps.append(precision); rs.append(recall)
            all_pred.append(pred); all_lab.append(rec["labels"])
        pooled_p, pooled_r, pooled_f1 = precision_recall_f1(
            np.concatenate(all_pred), np.concatenate(all_lab))
        summary[split + "_mean_f1"] = r(np.mean(f1s))
        summary[split + "_mean_p"] = r(np.mean(ps))
        summary[split + "_mean_r"] = r(np.mean(rs))
        summary[split + "_pooled"] = {"precision": r(pooled_p), "recall": r(pooled_r), "f1": r(pooled_f1)}
    return {
        "method": method, "model": model, "dataset": dataset,
        "fitted_params": fitted_params, "per_scene": per_scene, "summary": summary,
    }


# ---------------------------------------------------------------------------
# The six methods. Each returns the assembled JSON dict.
# ---------------------------------------------------------------------------

def run_fixed_05(model, dataset, val, test):
    """tau = 0.5 on raw scores, no fitting."""
    predict_fn = lambda rec: rec["scores"] > 0.5
    tau_of_scene = lambda rec: 0.5
    return assemble(model, "fixed_05", dataset, val, test, predict_fn, tau_of_scene,
                    {"tau": 0.5, "scale": "score"})


def run_f1_optimal(model, dataset, val, test):
    """One global tau swept on val (mean per-scene F1), applied to test."""
    tau, val_f1 = sweep_global_tau(val, lambda rec: rec["scores"])
    predict_fn = lambda rec: rec["scores"] > tau
    tau_of_scene = lambda rec: r(tau)
    result = assemble(model, "f1_optimal", dataset, val, test, predict_fn, tau_of_scene,
                      {"tau": r(tau), "scale": "score", "val_fit_mean_f1": r(val_f1)})
    return result, tau


def temp_prob(logits, T):
    """Calibrated P(change) = softmax(logits / T)[:, 1], computed stably."""
    z = logits / T                                       # divide both class logits by temperature
    z = z - z.max(axis=1, keepdims=True)                 # subtract row max for exp stability
    e = np.exp(z)
    return e[:, 1] / e.sum(axis=1)


def run_temperature(model, dataset, val, test):
    """Fit scalar T by min NLL on pooled val logits, then an F1-optimal tau on the calibrated probs."""
    val_logits = np.concatenate([rec["logits"] for rec in val])
    val_labels = np.concatenate([rec["labels"] for rec in val])

    def nll(T):
        p = np.column_stack([1.0 - temp_prob(val_logits, T), temp_prob(val_logits, T)])
        picked = p[np.arange(len(val_labels)), val_labels]   # probability of the true class
        return -np.mean(np.log(picked + 1e-12))              # mean negative log-likelihood

    T = float(minimize_scalar(nll, bounds=(0.05, 20.0), method="bounded").x)
    tau, val_f1 = sweep_global_tau(val, lambda rec: temp_prob(rec["logits"], T))
    # A monotonic rescale collapses to ONE raw-score threshold: temp_prob > tau <=> score > s_eff.
    s_eff = float(1.0 / (1.0 + np.exp(-(T * np.log(tau / (1.0 - tau))))))
    predict_fn = lambda rec: temp_prob(rec["logits"], T) > tau
    tau_of_scene = lambda rec: r(tau)
    fitted = {"T": r(T), "tau": r(tau), "scale": "calibrated_prob",
              "equivalent_score_threshold": r(s_eff), "val_fit_mean_f1": r(val_f1)}
    return assemble(model, "temperature_scaling", dataset, val, test, predict_fn, tau_of_scene, fitted)


def change_logodds(rec):
    """z = logit[:, 1] - logit[:, 0], the model's change log-odds (Platt input)."""
    return rec["logits"][:, 1] - rec["logits"][:, 0]


def run_platt(model, dataset, val, test):
    """A, B fit by logistic regression of change log-odds vs labels; decide at fixed 0.5."""
    z = np.concatenate([change_logodds(rec) for rec in val]).reshape(-1, 1)
    y = np.concatenate([rec["labels"] for rec in val])
    # C huge -> effectively unregularized, the textbook Platt MLE fit.
    clf = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    clf.fit(z, y)
    A = float(clf.coef_[0, 0])
    B = float(clf.intercept_[0])
    # Decision sigmoid(A*z + B) > 0.5  <=>  A*z + B > 0; report the implied score threshold.
    score_thresh = float(1.0 / (1.0 + np.exp(B / A))) if A != 0 else 0.5  # sigmoid(-B/A)
    predict_fn = lambda rec: (A * change_logodds(rec) + B) > 0.0
    tau_of_scene = lambda rec: 0.5
    fitted = {"A": r(A, 6), "B": r(B, 6), "scale": "calibrated_prob", "decision_threshold": 0.5,
              "equivalent_score_threshold": r(score_thresh)}
    return assemble(model, "platt", dataset, val, test, predict_fn, tau_of_scene, fitted)


def run_isotonic(model, dataset, val, test):
    """Monotonic score->label map fit on pooled val; decide at fixed 0.5."""
    s = np.concatenate([rec["scores"] for rec in val])
    y = np.concatenate([rec["labels"] for rec in val]).astype(np.float64)
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(s, y)
    # Implied raw-score threshold: smallest score whose mapped prob exceeds 0.5.
    grid = np.linspace(0.0, 1.0, 1001)
    mapped = iso.predict(grid)
    above = np.where(mapped > 0.5)[0]
    score_thresh = float(grid[above[0]]) if len(above) > 0 else 1.0
    # Store the compact PAV knots (downsample if very long) for reproducibility.
    knots_x = np.asarray(iso.X_thresholds_, dtype=np.float64)
    knots_y = np.asarray(iso.y_thresholds_, dtype=np.float64)
    if len(knots_x) > 1001:                              # keep the JSON small
        idx = np.linspace(0, len(knots_x) - 1, 1001).astype(int)
        knots_x, knots_y = knots_x[idx], knots_y[idx]
    predict_fn = lambda rec: iso.predict(rec["scores"]) > 0.5
    tau_of_scene = lambda rec: 0.5
    fitted = {"scale": "calibrated_prob", "decision_threshold": 0.5,
              "equivalent_score_threshold": r(score_thresh), "n_knots": int(len(iso.X_thresholds_)),
              "knots_x": [r(v) for v in knots_x], "knots_y": [r(v) for v in knots_y]}
    return assemble(model, "isotonic", dataset, val, test, predict_fn, tau_of_scene, fitted)


def cube_stats(rec):
    """Per-cube no-change mean/std/count of SCORES. No-change subset = score <= 0.5 (RAW
    model call, NO labels - Section 14). Returns arrays indexed by cube 0..K-1 plus the
    point->cube index. Computed with bincount, so it is vectorized and fast."""
    cube_id = rec["cube_id"]
    scores = rec["scores"]
    cubes, inv = np.unique(cube_id, return_inverse=True)  # inv[p] = index of point p's cube
    K = len(cubes)
    nochange = (scores <= 0.5).astype(np.float64)         # 1.0 where the model calls no-change
    count = np.bincount(inv, weights=nochange, minlength=K)            # no-change pts per cube
    ssum = np.bincount(inv, weights=nochange * scores, minlength=K)    # sum of those scores
    ssq = np.bincount(inv, weights=nochange * scores * scores, minlength=K)  # sum of squares
    safe = np.maximum(count, 1.0)                         # avoid divide-by-zero for empty cubes
    mean = ssum / safe
    var = np.maximum(ssq / safe - mean * mean, 0.0)       # population variance, clip tiny fp negatives
    std = np.sqrt(var)
    mean = np.where(count > 0, mean, 0.0)
    std = np.where(count > 0, std, 0.0)
    return {"count": count, "mean": mean, "std": std, "inv": inv, "K": K}


def two_moment_pred(stats, scores, c, lam, fb_tau):
    """Per-point change decision under tau_i = clip(c + mean_i + lam*std_i), with cubes
    that have fewer than MIN_NC no-change points falling back to the global f1_optimal tau."""
    tau_cube = np.clip(c + stats["mean"] + lam * stats["std"], 0.0, 1.0)   # two-moment tau per cube
    small = stats["count"] < MIN_NC                                        # unstable std cubes
    tau_cube = np.where(small, fb_tau, tau_cube)                           # documented fallback
    return scores > tau_cube[stats["inv"]]                                 # broadcast tau to points


def precompute_fast_search(rec, stats):
    """Precompute, ONCE per val scene, the structures that let the (c, lambda) grid search score
    a candidate in O(cubes) instead of O(points). For each cube we will need, given a per-cube
    threshold, how many of its points exceed it, split by label. We get that from per-cube sorted
    scores via searchsorted, vectorized across cubes with a key-offset trick: key = OFFSET*cube +
    score with OFFSET > 1 keeps cubes from interleaving (scores are in [0, 1]), so one global
    searchsorted answers every cube at once. The reported metrics still come from two_moment_pred
    (below); this only accelerates picking (c, lambda)."""
    scores = rec["scores"]
    labels = rec["labels"]
    inv = stats["inv"]                                   # point -> dense cube index 0..K-1
    K = stats["K"]
    OFFSET = 2.0                                          # > 1 so cube i occupies key band [2i, 2i+1]
    is_pos = labels == 1
    is_neg = ~is_pos
    # Per-cube sorted keys for positives and negatives separately.
    pos_key = np.sort(OFFSET * inv[is_pos] + scores[is_pos])   # globally sorted == per-cube sorted
    neg_key = np.sort(OFFSET * inv[is_neg] + scores[is_neg])
    n_pos = np.bincount(inv[is_pos], minlength=K).astype(np.int64)   # positives per cube
    n_neg = np.bincount(inv[is_neg], minlength=K).astype(np.int64)   # negatives per cube
    # Exclusive prefix sums = number of pos/neg points lying in cubes BEFORE cube i (= that cube's
    # start offset in the globally sorted key array).
    cum_pos = np.concatenate([[0], np.cumsum(n_pos)[:-1]])
    cum_neg = np.concatenate([[0], np.cumsum(n_neg)[:-1]])
    return {"pos_key": pos_key, "neg_key": neg_key, "n_pos": n_pos, "n_neg": n_neg,
            "cum_pos": cum_pos, "cum_neg": cum_neg, "total_pos": int(n_pos.sum()),
            "K": K, "OFFSET": OFFSET, "idx": np.arange(K),
            "mean": stats["mean"], "std": stats["std"], "count": stats["count"]}


def scene_f1_fast(pre, c, lam, fb_tau):
    """Change-class F1 for one scene under tau_i = clip(c + mean_i + lam*std_i) (with the MIN_NC
    fallback), computed from the precomputed structures. Replicates two_moment_pred + f1_from_pred
    EXACTLY: strict 'score > tau' via searchsorted side='right', same fallback, same F1 formula."""
    tau = np.clip(c + pre["mean"] + lam * pre["std"], 0.0, 1.0)   # per-cube two-moment threshold
    tau = np.where(pre["count"] < MIN_NC, fb_tau, tau)            # documented small-cube fallback
    query = pre["OFFSET"] * pre["idx"] + tau                      # one query key per cube
    # searchsorted 'right' counts keys <= query (i.e. score <= tau, so NOT predicted change). For
    # cube i this is (all points in earlier cubes) + (cube i points with score <= tau_i); subtract
    # the earlier-cube offset to isolate cube i, then n_i minus that is the count predicted change.
    pos_le = np.searchsorted(pre["pos_key"], query, side="right") - pre["cum_pos"]
    neg_le = np.searchsorted(pre["neg_key"], query, side="right") - pre["cum_neg"]
    tp = int((pre["n_pos"] - pos_le).sum())              # positives predicted change
    fp = int((pre["n_neg"] - neg_le).sum())              # negatives predicted change
    fn = pre["total_pos"] - tp                           # positives not predicted change
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0


def run_two_moment(model, dataset, val, test, fb_tau):
    """Fit (c, lambda) once on val by grid search (mean per-scene F1), apply per-cube on test."""
    stats_by_scene = {rec["scene"]: cube_stats(rec) for rec in (val + test)}

    # Fast path: precompute val structures once, then score each (c, lambda) in O(cubes). This is
    # an exact accelerator for the SELECTION only; every reported number still flows through
    # two_moment_pred below, so a hypothetical search bug could at worst pick a different (c, lambda).
    val_pre = [precompute_fast_search(rec, stats_by_scene[rec["scene"]]) for rec in val]
    best_c, best_lam, best_f1 = 0.0, 0.0, -1.0
    for c in C_GRID:
        for lam in LAM_GRID:
            mean_f1 = float(np.mean([scene_f1_fast(pre, c, lam, fb_tau) for pre in val_pre]))
            if mean_f1 > best_f1:
                best_f1, best_c, best_lam = mean_f1, float(c), float(lam)

    predict_fn = lambda rec: two_moment_pred(stats_by_scene[rec["scene"]], rec["scores"], best_c, best_lam, fb_tau)
    tau_of_scene = lambda rec: [r(v) for v in np.clip(
        np.where(stats_by_scene[rec["scene"]]["count"] < MIN_NC, fb_tau,
                 best_c + stats_by_scene[rec["scene"]]["mean"] + best_lam * stats_by_scene[rec["scene"]]["std"]),
        0.0, 1.0)]
    fitted = {"c": r(best_c), "lambda": r(best_lam), "min_nc": MIN_NC,
              "fallback": "global_f1_optimal_tau", "fallback_tau": r(fb_tau),
              "nochange_subset": "score<=0.5 (raw prediction, no labels)",
              "val_fit_mean_f1": r(best_f1)}
    result = assemble(model, "two_moment", dataset, val, test, predict_fn, tau_of_scene, fitted)
    # Record per-scene cube counts and how many cubes hit the fallback (0 on LD; matters for IndoorCD).
    for rec in (val + test):
        stats = stats_by_scene[rec["scene"]]
        result["per_scene"][rec["scene"]]["n_cubes"] = int(stats["K"])
        result["per_scene"][rec["scene"]]["n_fallback"] = int(np.sum(stats["count"] < MIN_NC))
    return result, best_c, best_lam


def load_diag(dataset):
    """From results/<dataset>_discrimination.csv read the per-cube oracle ceiling (mean of
    test per-scene percube_oracle_pooled) and the pooled val best_global_f1 (the f1_optimal gate)."""
    path = os.path.join("results", dataset + "_discrimination.csv")
    oracle, bestg_val = {}, {}
    if not os.path.exists(path):
        return {}, {}
    with open(path) as f:
        for row in csv.DictReader(f):
            m = row["model"]
            if row["scene"] != "POOLED" and row["split"] == "test" and row["percube_oracle_pooled"]:
                oracle.setdefault(m, []).append(float(row["percube_oracle_pooled"]))
            if row["scene"] == "POOLED" and row["split"] == "val":
                bestg_val[m] = float(row["best_global_f1"])
    ceiling = {m: float(np.mean(v)) for m, v in oracle.items()}
    return ceiling, bestg_val


# Fixed method order for the table and the JSON files.
METHODS = ["fixed_05", "f1_optimal", "temperature_scaling", "platt", "isotonic", "two_moment"]


def main():
    parser = argparse.ArgumentParser(description="Run all six calibration methods on saved predictions.")
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
    ceiling, bestg_val = load_diag(dataset)

    table_rows = []
    tm_rows = []
    for model in models:
        val, test = load_split(os.path.join(pred_root, model), split_of)
        print("\n== %s ==  (val %d scenes, test %d scenes)" % (model, len(val), len(test)))

        # Run the six methods. f1_optimal's tau is reused as the two_moment fallback.
        results = {}
        results["fixed_05"] = run_fixed_05(model, dataset, val, test)
        results["f1_optimal"], f1opt_tau = run_f1_optimal(model, dataset, val, test)
        results["temperature_scaling"] = run_temperature(model, dataset, val, test)
        results["platt"] = run_platt(model, dataset, val, test)
        results["isotonic"] = run_isotonic(model, dataset, val, test)
        results["two_moment"], tm_c, tm_lam = run_two_moment(model, dataset, val, test, f1opt_tau)

        # Write the per-method JSON files (Section 8 sits beside this in predictions/).
        out_dir = os.path.join("calibration", dataset, model)
        os.makedirs(out_dir, exist_ok=True)
        for method in METHODS:
            with open(os.path.join(out_dir, method + ".json"), "w") as f:
                json.dump(results[method], f, indent=2)

        # Collect the table rows and print a compact per-model summary.
        print("  %-20s %8s %8s   %8s %8s" % ("method", "valF1", "testF1", "valPool", "testPool"))
        for method in METHODS:
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
            print("  %-20s %8.4f %8.4f   %8.4f %8.4f" % (
                method, s["val_mean_f1"], s["test_mean_f1"], s["val_pooled"]["f1"], s["test_pooled"]["f1"]))
        tm_rows.append({"dataset": dataset, "model": model, "c": r(tm_c), "lambda": r(tm_lam),
                        "min_nc": MIN_NC, "fallback_tau": r(f1opt_tau)})

        # ---- Sanity gates (Section 14 / Task 7 contract) ----
        f5 = results["fixed_05"]["summary"]["val_mean_f1"]
        fo = results["f1_optimal"]["summary"]["val_mean_f1"]
        assert fo >= f5 - 1e-9, "f1_optimal val F1 below fixed_05 (impossible if 0.5 is in the grid)"
        if model in bestg_val:
            print("  gate f1_optimal val F1 %.4f  vs diagnostic pooled best_global %.4f" % (fo, bestg_val[model]))
        if model in ceiling:
            cap = ceiling[model]
            print("  per-cube oracle ceiling (test, mean per-scene) = %.4f" % cap)
            for method in METHODS:
                tf = results[method]["summary"]["test_mean_f1"]
                if tf > cap + 0.02:                       # log-only: a method should not beat the per-cube oracle
                    print("  WARN %s test F1 %.4f exceeds oracle ceiling %.4f" % (method, tf, cap))
        # Honest collapse check: temperature reduces to one score threshold ~ f1_optimal.
        t_eff = results["temperature_scaling"]["fitted_params"]["equivalent_score_threshold"]
        t_test = results["temperature_scaling"]["summary"]["test_mean_f1"]
        print("  temperature: T=%.3f  equiv score thr=%.3f (f1_optimal tau=%.3f)  test F1 %.4f vs f1_optimal %.4f"
              % (results["temperature_scaling"]["fitted_params"]["T"], t_eff, f1opt_tau,
                 t_test, results["f1_optimal"]["summary"]["test_mean_f1"]))
        print("  two_moment: c=%.3f lambda=%.3f  test F1 %.4f" % (tm_c, tm_lam, results["two_moment"]["summary"]["test_mean_f1"]))

    # Write the main table (model x method) and the two_moment params (critical for Task 10).
    os.makedirs("results", exist_ok=True)
    table_fields = ["dataset", "model", "method", "val_mean_p", "val_mean_r", "val_mean_f1",
                    "test_mean_p", "test_mean_r", "test_mean_f1", "val_pool_p", "val_pool_r",
                    "val_pool_f1", "test_pool_p", "test_pool_r", "test_pool_f1"]
    # UPSERT into the single main_table.csv (Section 9 single-table contract): keep every
    # other dataset's rows, drop this dataset's old rows, then append the freshly computed
    # ones. Without this, opening in "w" mode would clobber the LD rows when calibrate runs
    # on MS (the rows are keyed by dataset, so re-running one dataset is idempotent).
    main_table_path = os.path.join("results", "main_table.csv")
    kept_rows = []
    if os.path.exists(main_table_path):
        with open(main_table_path, newline="") as f:
            for row in csv.DictReader(f):
                if row.get("dataset") != dataset:        # other datasets survive untouched
                    kept_rows.append(row)
    with open(main_table_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=table_fields, extrasaction="ignore")
        w.writeheader()
        for row in kept_rows:                            # other datasets first
            w.writerow(row)
        for row in table_rows:                           # then this dataset's new rows
            w.writerow(row)
    with open(os.path.join("results", "two_moment_params_" + dataset + ".csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "model", "c", "lambda", "min_nc", "fallback_tau"])
        w.writeheader()
        for row in tm_rows:
            w.writerow(row)

    assert len(table_rows) == len(models) * len(METHODS), "main table is missing model x method rows"
    print("\nupserted %d %s rows into results/main_table.csv (%d models x %d methods; kept %d rows from other datasets)" % (
        len(table_rows), dataset, len(models), len(METHODS), len(kept_rows)))
    print("wrote results/two_moment_params_%s.csv" % dataset)
    print("wrote calibration/%s/<model>/<method>.json (%d files)" % (dataset, len(models) * len(METHODS)))

    # Headline: did two_moment beat f1_optimal, and how much of the per-cube headroom did it capture.
    print("\n== HEADLINE (test mean per-scene change-F1) ==")
    print("  %-20s %8s %8s %8s %8s %8s" % ("model", "f1opt", "two_mom", "gain", "ceiling", "head%"))
    for row in tm_rows:
        model = row["model"]
        fo = next(t for t in table_rows if t["model"] == model and t["method"] == "f1_optimal")["test_mean_f1"]
        tm = next(t for t in table_rows if t["model"] == model and t["method"] == "two_moment")["test_mean_f1"]
        cap = ceiling.get(model, float("nan"))
        headroom = cap - fo
        captured = (tm - fo) / headroom * 100.0 if headroom > 1e-9 else float("nan")
        print("  %-20s %8.4f %8.4f %+8.4f %8.4f %7.1f%%" % (model, fo, tm, tm - fo, cap, captured))


if __name__ == "__main__":
    main()
