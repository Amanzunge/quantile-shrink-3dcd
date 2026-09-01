"""
paper_stats.py - Pre-submission evidence tables: change-class IoU + significance.

Builds two tables from the FROZEN predictions and the stored calibration
artefacts. Nothing frozen is touched; the only refit is the isotonic map on val,
which is deterministic and reproduces the Task 7 fit exactly.

  results/iou_table.csv
      change-class IoU (TP / (TP+FP+FN)), mean per-scene and pooled, on val and
      test, for the six paper methods + the otsu_cube baseline. IoU is what the
      Urb3DCD literature reports, so this makes our numbers comparable to it.

  results/significance_table.csv
      headline comparisons quantile_shrink vs f1_optimal and vs otsu_cube:
      paired bootstrap 95% CI of the difference in test mean per-scene change-F1
      (resampling test cubes within each scene, same draws for both methods),
      a two-sided bootstrap p-value, and a per-scene Wilcoxon signed-rank test
      where the dataset has enough test scenes. Final ALL_COMBOS rows pool the
      22 model x dataset combos (Wilcoxon over the combo-level differences).

Every method's decision rule is REBUILT from its stored fitted parameters and
VERIFIED against the per-scene F1 recorded in its calibration JSON (tolerance
3e-3) before being used, so these are exactly the paper numbers.

Run calibrate_otsu.py for all datasets first (it writes the otsu_cube JSONs).

Torch-free, local CPU:  python src/paper_stats.py
"""

import os
import glob
import json
import csv
import argparse
import numpy as np
import yaml
from scipy.stats import wilcoxon
from sklearn.isotonic import IsotonicRegression

from calibrate import r
from calibrate_v2 import ALPHA_GRID, precompute_scene, tau_quantile_shrink
from calibrate_otsu import otsu_scene

# The six paper methods (main_table.csv) plus the new reviewer baseline.
METHODS = ["fixed_05", "f1_optimal", "temperature_scaling", "platt", "isotonic",
           "quantile_shrink", "otsu_cube"]
# Methods that live in calibration_v2/ instead of calibration/.
V2_METHODS = {"quantile_shrink", "otsu_cube"}
# The significance analysis compares the headline method against these two.
SIG_METHODS = ["quantile_shrink", "f1_optimal", "otsu_cube"]
PAIRS = [("quantile_shrink", "f1_optimal"), ("quantile_shrink", "otsu_cube")]

B_BOOT = 10000        # bootstrap resamples
SEED = 42             # project seed (Section 7)
VERIFY_TOL = 3e-3     # max allowed |recomputed - stored| per-scene F1 (stored is 4dp-rounded)
WILCOXON_MIN = 5      # fewer test scenes than this -> per-scene Wilcoxon not reported

CONFIGS = ["configs/urb3dcd_v2_ld.yaml", "configs/urb3dcd_v2_ms.yaml",
           "configs/hkcd.yaml", "configs/indoorcd.yaml"]


def load_scenes(pred_dir, split_of):
    """Val and test scene records with the change log-odds z (platt/temperature input)
    instead of the full logits, to keep the big HKCD scenes affordable."""
    val, test = [], []
    for path in sorted(glob.glob(os.path.join(pred_dir, "*.npz"))):
        d = np.load(path, allow_pickle=True)
        scene = str(d["scene_id"])
        split = split_of.get(scene, "?")
        if split not in ("val", "test"):
            continue
        logits = d["logits"].astype(np.float64)
        rec = {"scene": scene,
               "scores": d["scores"].astype(np.float64),
               "z": logits[:, 1] - logits[:, 0],
               "labels": d["labels"].astype(np.int64),
               "cube_id": d["cube_id"]}
        (val if split == "val" else test).append(rec)
    return val, test


def load_artefacts(dataset, model):
    """The stored calibration JSON of every method for one dataset x model."""
    art = {}
    for m in METHODS:
        root = "calibration_v2" if m in V2_METHODS else "calibration"
        with open(os.path.join(root, dataset, model, m + ".json")) as f:
            art[m] = json.load(f)
    return art


def build_ctx(art, val, test):
    """Reconstruct every decision rule from the stored fitted parameters:
    scalars for the global methods, a deterministic isotonic refit on val, and
    per-scene per-cube tau vectors for quantile_shrink and otsu_cube."""
    ctx = {}
    ctx["tau_f1opt"] = float(art["f1_optimal"]["fitted_params"]["tau"])
    ctx["T"] = float(art["temperature_scaling"]["fitted_params"]["T"])
    ctx["tau_T"] = float(art["temperature_scaling"]["fitted_params"]["tau"])
    ctx["A"] = float(art["platt"]["fitted_params"]["A"])
    ctx["B"] = float(art["platt"]["fitted_params"]["B"])

    # Isotonic refit on pooled val, identical to calibrate.run_isotonic (deterministic).
    s = np.concatenate([rec["scores"] for rec in val])
    y = np.concatenate([rec["labels"] for rec in val]).astype(np.float64)
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(s, y)
    ctx["iso"] = iso

    # quantile_shrink per-cube taus from the stored (c, alpha, k).
    fp = art["quantile_shrink"]["fitted_params"]
    ai = int(np.argmin(np.abs(ALPHA_GRID - float(fp["alpha"]))))
    assert abs(ALPHA_GRID[ai] - float(fp["alpha"])) < 1e-9, "stored alpha not on the grid"
    c, k = float(fp["c"]), float(fp["k"])

    ctx["inv"], ctx["tau_qs"], ctx["tau_otsu"] = {}, {}, {}
    for rec in (val + test):
        pre = precompute_scene(rec)                  # calibrate_v2 machinery, exact
        ctx["inv"][rec["scene"]] = pre["inv"]
        ctx["tau_qs"][rec["scene"]] = tau_quantile_shrink(pre, c, ai, k)
        tau_ot, inv_ot, _ = otsu_scene(rec)          # calibrate_otsu machinery, exact
        assert np.array_equal(inv_ot, pre["inv"]), "cube index mismatch between machineries"
        ctx["tau_otsu"][rec["scene"]] = tau_ot
    return ctx


def build_masks(rec, ctx):
    """All seven per-point change decisions for one scene, from the rebuilt rules."""
    s, z = rec["scores"], rec["z"]
    inv = ctx["inv"][rec["scene"]]
    masks = {}
    masks["fixed_05"] = s > 0.5
    masks["f1_optimal"] = s > ctx["tau_f1opt"]
    # temp_prob(logits, T) > tau  <=>  z > T * logit(tau); exact monotonic identity.
    masks["temperature_scaling"] = z > ctx["T"] * np.log(ctx["tau_T"] / (1.0 - ctx["tau_T"]))
    masks["platt"] = (ctx["A"] * z + ctx["B"]) > 0.0
    masks["isotonic"] = ctx["iso"].predict(s) > 0.5
    masks["quantile_shrink"] = s > ctx["tau_qs"][rec["scene"]][inv]
    masks["otsu_cube"] = s > ctx["tau_otsu"][rec["scene"]][inv]
    return masks


def f1_of(tp, fp, fn):
    """Change-class F1 from counts; 2PR/(P+R) == 2tp/(2tp+fp+fn)."""
    den = 2 * tp + fp + fn
    return 2 * tp / den if den > 0 else 0.0


def iou_of(tp, fp, fn):
    """Change-class IoU from counts; 0 when the class is absent on both sides,
    matching the F1 zero convention of Task 7."""
    den = tp + fp + fn
    return tp / den if den > 0 else 0.0


def f1_rows(counts):
    """Vectorized change-F1 over rows of an (B, 3) tp/fp/fn count array."""
    tp, fp, fn = counts[:, 0], counts[:, 1], counts[:, 2]
    den = 2 * tp + fp + fn
    return np.where(den > 0, 2 * tp / np.maximum(den, 1), 0.0)


def paired_bootstrap(counts_a, counts_b, n_boot, seed):
    """Paired bootstrap of the difference in mean per-scene change-F1. counts_x is a
    list over test scenes of (K, 3) per-cube tp/fp/fn arrays; cubes are resampled
    with replacement WITHIN each scene, with the same draws for both methods.
    Returns (ci_lo, ci_hi, p_two_sided)."""
    rng = np.random.default_rng(seed)
    n_scenes = len(counts_a)
    f1_a = np.zeros((n_scenes, n_boot))
    f1_b = np.zeros((n_scenes, n_boot))
    for si in range(n_scenes):
        n_cubes = counts_a[si].shape[0]
        idx = rng.integers(0, n_cubes, size=(n_boot, n_cubes))   # shared draws = paired
        f1_a[si] = f1_rows(counts_a[si][idx].sum(axis=1))
        f1_b[si] = f1_rows(counts_b[si][idx].sum(axis=1))
    delta = f1_a.mean(axis=0) - f1_b.mean(axis=0)
    ci_lo, ci_hi = np.percentile(delta, [2.5, 97.5])
    # Two-sided sign p-value with the standard +1 correction, capped at 1.
    p_lo = (1 + np.sum(delta <= 0.0)) / (n_boot + 1)
    p_hi = (1 + np.sum(delta >= 0.0)) / (n_boot + 1)
    return float(ci_lo), float(ci_hi), float(min(2.0 * min(p_lo, p_hi), 1.0))


def process_model(dataset, model, split_of):
    """One dataset x model: rebuild all decisions, verify against the stored JSONs,
    return IoU rows plus the significance inputs (per-cube test counts and
    per-scene test F1 per SIG method)."""
    val, test = load_scenes(os.path.join("predictions", dataset, model), split_of)
    art = load_artefacts(dataset, model)
    ctx = build_ctx(art, val, test)

    scene_counts = {m: [] for m in METHODS}          # (split, scene, tp, fp, fn)
    cube_counts = {m: [] for m in SIG_METHODS}       # per test scene: (K, 3)
    scene_f1_test = {m: [] for m in SIG_METHODS}     # per test scene: change-F1
    max_diff = 0.0

    for rec, split in [(rec, "val") for rec in val] + [(rec, "test") for rec in test]:
        masks = build_masks(rec, ctx)
        is_pos = rec["labels"] == 1
        n_pos = int(is_pos.sum())
        inv = ctx["inv"][rec["scene"]]
        n_cubes = len(ctx["tau_qs"][rec["scene"]])
        for m in METHODS:
            mask = masks[m]
            tp = int(np.sum(mask & is_pos))
            fp = int(mask.sum()) - tp
            fn = n_pos - tp
            scene_counts[m].append((split, rec["scene"], tp, fp, fn))
            # Verification gate: the rebuilt rule must reproduce the stored number.
            diff = abs(f1_of(tp, fp, fn) - art[m]["per_scene"][rec["scene"]]["f1"])
            max_diff = max(max_diff, diff)
            if split == "test" and m in SIG_METHODS:
                tp_c = np.bincount(inv, weights=(mask & is_pos), minlength=n_cubes)
                fp_c = np.bincount(inv, weights=(mask & ~is_pos), minlength=n_cubes)
                pos_c = np.bincount(inv, weights=is_pos, minlength=n_cubes)
                cube_counts[m].append(
                    np.stack([tp_c, fp_c, pos_c - tp_c], axis=1).astype(np.int64))
                scene_f1_test[m].append(f1_of(tp, fp, fn))
        del masks

    assert max_diff <= VERIFY_TOL, \
        "%s/%s: rebuilt decisions drift from stored JSONs (max F1 diff %.5f)" % (dataset, model, max_diff)

    # Cross-check: per-cube counts must sum back to the scene totals.
    for m in SIG_METHODS:
        totals = [(t, f, n) for sp, _, t, f, n in scene_counts[m] if sp == "test"]
        for (t, f, n), cc in zip(totals, cube_counts[m]):
            assert (t, f, n) == tuple(cc.sum(axis=0)), "cube counts do not sum to scene counts"

    # IoU rows (mean per-scene + pooled, val and test) for all seven methods.
    iou_rows = []
    for m in METHODS:
        row = {"dataset": dataset, "model": model, "method": m}
        for split in ["val", "test"]:
            triples = [(t, f, n) for sp, _, t, f, n in scene_counts[m] if sp == split]
            row[split + "_mean_iou"] = r(np.mean([iou_of(*t) for t in triples]))
            pooled = tuple(np.sum(triples, axis=0))
            row[split + "_pool_iou"] = r(iou_of(*pooled))
            row[split + "_mean_f1"] = r(np.mean([f1_of(*t) for t in triples]))
        iou_rows.append(row)

    print("  %-20s verified (max per-scene F1 drift %.5f)" % (model, max_diff))
    return iou_rows, cube_counts, scene_f1_test, len(test)


def main():
    parser = argparse.ArgumentParser(description="Build the IoU and significance tables.")
    parser.add_argument("--configs", nargs="*", default=CONFIGS,
                        help="dataset YAMLs (default: all four; partial runs rewrite the tables)")
    args = parser.parse_args()

    iou_rows = []
    sig_rows = []
    combo_deltas = {pair: [] for pair in PAIRS}      # combo-level test mean-F1 differences

    for config in args.configs:
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
            rows, cube_counts, scene_f1, n_test = process_model(dataset, model, split_of)
            iou_rows.extend(rows)

            for m_a, m_b in PAIRS:
                mean_a = float(np.mean(scene_f1[m_a]))
                mean_b = float(np.mean(scene_f1[m_b]))
                ci_lo, ci_hi, boot_p = paired_bootstrap(cube_counts[m_a], cube_counts[m_b],
                                                        B_BOOT, SEED)
                diffs = np.array(scene_f1[m_a]) - np.array(scene_f1[m_b])
                nonzero = int(np.count_nonzero(diffs))
                if n_test >= WILCOXON_MIN and nonzero > 0:
                    w_p = r(float(wilcoxon(diffs).pvalue), 6)
                else:
                    w_p = ""                          # too few scenes (or no differences) to test
                combo_deltas[(m_a, m_b)].append(mean_a - mean_b)
                sig_rows.append({
                    "dataset": dataset, "model": model,
                    "comparison": m_a + "_vs_" + m_b,
                    "n_test_scenes": n_test,
                    "n_test_cubes": int(sum(len(c) for c in cube_counts[m_a])),
                    "f1_a": r(mean_a), "f1_b": r(mean_b), "delta": r(mean_a - mean_b),
                    "boot_ci_lo": r(ci_lo), "boot_ci_hi": r(ci_hi), "boot_p": r(boot_p, 6),
                    "wilcoxon_nonzero_n": nonzero, "wilcoxon_p": w_p,
                })

    # Pooled test over the 22 model x dataset combos (the Demsar-style global check).
    for m_a, m_b in PAIRS:
        deltas = np.array(combo_deltas[(m_a, m_b)])
        w_p = r(float(wilcoxon(deltas).pvalue), 6) if np.count_nonzero(deltas) > 0 else ""
        sig_rows.append({
            "dataset": "ALL_COMBOS", "model": "%d combos" % len(deltas),
            "comparison": m_a + "_vs_" + m_b,
            "n_test_scenes": "", "n_test_cubes": "",
            "f1_a": "", "f1_b": "", "delta": r(float(deltas.mean())),
            "boot_ci_lo": "", "boot_ci_hi": "", "boot_p": "",
            "wilcoxon_nonzero_n": int(np.count_nonzero(deltas)), "wilcoxon_p": w_p,
        })
        wins = int(np.sum(deltas > 0))
        losses = int(np.sum(deltas < 0))
        print("\n%s vs %s over %d combos: mean delta %+.4f, wins %d / losses %d, Wilcoxon p %s"
              % (m_a, m_b, len(deltas), deltas.mean(), wins, losses, w_p))

    os.makedirs("results", exist_ok=True)
    iou_fields = ["dataset", "model", "method", "val_mean_iou", "val_pool_iou", "val_mean_f1",
                  "test_mean_iou", "test_pool_iou", "test_mean_f1"]
    with open(os.path.join("results", "iou_table.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=iou_fields)
        w.writeheader()
        for row in iou_rows:
            w.writerow(row)

    sig_fields = ["dataset", "model", "comparison", "n_test_scenes", "n_test_cubes",
                  "f1_a", "f1_b", "delta", "boot_ci_lo", "boot_ci_hi", "boot_p",
                  "wilcoxon_nonzero_n", "wilcoxon_p"]
    with open(os.path.join("results", "significance_table.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sig_fields)
        w.writeheader()
        for row in sig_rows:
            w.writerow(row)

    print("\nwrote results/iou_table.csv (%d rows) and results/significance_table.csv (%d rows)"
          % (len(iou_rows), len(sig_rows)))


if __name__ == "__main__":
    main()
