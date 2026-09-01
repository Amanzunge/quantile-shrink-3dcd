"""
transfer.py - Task 10. Cross-dataset transfer of the two-moment (c, lambda).

This directly tests CLAUDE Section 0's central claim: the TWO scalars (c, lambda) of the
two-moment per-cube threshold transfer across datasets WITHOUT per-dataset refitting.

  tau_i = clip[0,1]( c + mean_nochange_i + lambda * std_nochange_i )

We take (c, lambda) fit on a SOURCE dataset's validation set (already saved by calibrate.py
in results/two_moment_params_<source>.csv) and apply them UNCHANGED to a TARGET dataset's
TEST predictions, per-cube, then compare:
  (a) transferred  : SOURCE (c, lambda) on the TARGET test            <- the claim
  (b) refit        : TARGET (c, lambda) on the TARGET test            <- the per-dataset tuned version
  references       : fixed_05, f1_optimal (TARGET-tuned), per-cube oracle ceiling

The headline is the transfer GAP (b - a) and whether the transferred (a) still matches/beats
the per-dataset-tuned f1_optimal (the "calibrate once, reuse everywhere with no retuning" claim).

Only (c, lambda) come from the source. The per-cube no-change mean/std are computed at TEST
TIME from the TARGET predictions (Section 0 / Section 14), so the per-cube statistics adapt to
the target automatically; the transfer is purely the two global scalars.

Reuses calibrate.py's EXACT per-cube math (cube_stats, two_moment_pred) and the same
diagnose_predictions.f1_from_pred metric, so transfer numbers are directly comparable to the
main table. Torch-free, no GPU, no retraining. Local CPU analysis (Section 12).

  python src/transfer.py

Run from the repo root (paths under results/, predictions/ are relative, like calibrate.py).
"""

import os
import csv
import argparse
import numpy as np
import yaml

# Reuse the locked per-cube logic and loaders from Task 7's calibrate.py so nothing is
# reinvented: cube_stats (no-change subset = score<=0.5 raw, per-cube mean/std/count),
# two_moment_pred (tau_i = clip(c + mean_i + lam*std_i) with the MIN_NC small-cube fallback),
# load_split (per-scene npz -> val/test recs), load_diag (per-cube oracle ceiling).
from calibrate import cube_stats, two_moment_pred, load_split, load_diag, MIN_NC
from diagnose_predictions import f1_from_pred


# Config file per dataset, so we can map scene_id -> split exactly like calibrate.py does.
CONFIG_OF = {
    "urb3dcd_v2_ld": "configs/urb3dcd_v2_ld.yaml",
    "urb3dcd_v2_ms": "configs/urb3dcd_v2_ms.yaml",
    "hkcd": "configs/hkcd.yaml",
}

# Models that FAILED the AUC discrimination gate on a dataset (recorded in the Task 6/8/9 notes
# and visible as boundary-pinned (c, lambda) in the params CSVs). A transfer is only meaningful
# when BOTH ends are healthy: the SOURCE model's (c, lambda) must be a genuine interior optimum,
# and the TARGET model's predictions must be discriminative. Failed ends are reported as N/A and
# kept OUT of the headline so they do not distort it.
#   MS  siamese_pointnet : collapsed on MS (AUC ~0.56), boundary c=-0.2
#   HKCD siamese_kpconv  : failed on HKCD (AUC 0.39/0.57), boundary c=-0.18
FAILED = {
    "urb3dcd_v2_ld": set(),                  # all 5 LD models AUC-healthy (Task 5/6)
    "urb3dcd_v2_ms": {"siamese_pointnet"},
    "hkcd": {"siamese_kpconv"},
}


def r(x, n=4):
    """Round to a plain python float for the CSV (matches calibrate.py rounding)."""
    return round(float(x), n)


def split_of_dataset(dataset):
    """scene_id -> 'train'/'val'/'test' for one dataset, from its frozen splits.json."""
    import json
    cfg = yaml.safe_load(open(CONFIG_OF[dataset]))
    splits = json.load(open(cfg["splits_file"]))
    mapping = {}
    for split in ["train", "val", "test"]:
        for rel in splits.get(split, []):
            mapping[rel.split("/")[-1]] = split          # last path component is the scene_id
    return mapping


def load_two_moment_params(dataset):
    """results/two_moment_params_<dataset>.csv -> {model: {'c','lambda','fallback_tau'}}."""
    path = os.path.join("results", "two_moment_params_" + dataset + ".csv")
    out = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            out[row["model"]] = {
                "c": float(row["c"]),
                "lambda": float(row["lambda"]),
                "fallback_tau": float(row["fallback_tau"]),
            }
    return out


def load_main_table_refs(dataset):
    """results/main_table.csv -> {(model, method): test_mean_f1} for this dataset.
    Gives the already-computed fixed_05 / f1_optimal / two_moment references for free."""
    path = os.path.join("results", "main_table.csv")
    out = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row["dataset"] == dataset:
                out[(row["model"], row["method"])] = float(row["test_mean_f1"])
    return out


def load_target_test(dataset, model):
    """Load this model's TEST scene recs for the target dataset and precompute per-cube stats.
    Returns (test_recs, stats_by_scene). Stats use the TARGET's own test-time predictions."""
    split_of = split_of_dataset(dataset)
    pred_dir = os.path.join("predictions", dataset, model)
    _val, test = load_split(pred_dir, split_of)            # we only need the test scenes here
    stats_by_scene = {rec["scene"]: cube_stats(rec) for rec in test}
    return test, stats_by_scene


def mean_test_f1(test_recs, stats_by_scene, c, lam, fb_tau):
    """Mean over test scenes of per-scene change-F1 under the two-moment per-cube threshold
    with the given (c, lambda) and small-cube fallback tau. Identical to how calibrate.py
    scores two_moment on test, just with a (c, lambda) we choose."""
    f1s = []
    for rec in test_recs:
        pred = two_moment_pred(stats_by_scene[rec["scene"]], rec["scores"], c, lam, fb_tau)
        f1s.append(f1_from_pred(pred, rec["labels"]))
    return float(np.mean(f1s))


def main():
    parser = argparse.ArgumentParser(description="Task 10: cross-dataset transfer of two-moment (c, lambda).")
    parser.add_argument("--out", default=os.path.join("results", "transfer_table.csv"),
                        help="output CSV path")
    args = parser.parse_args()

    # The transfer experiments. is_primary marks the headline (sim->real into HKCD, per CLAUDE
    # Section 10): LD->HKCD for the 5 LD models, and MS->HKCD for randla (which has NO LD twin,
    # Section 2 excludes randla from LD). The secondary blocks (MS->HKCD for the LD models, and
    # LD->MS) show the transfer is not a single lucky pair.
    specs = []
    ld_models = ["icp_euclidean", "siamese_pointnet", "siamese_pointnet2", "siamese_kpconv", "siamgcn"]
    for m in ld_models:                                    # PRIMARY: simulated LD -> real HKCD
        specs.append(("urb3dcd_v2_ld", "hkcd", m, True))
    specs.append(("urb3dcd_v2_ms", "hkcd", "randla", True))   # PRIMARY for randla (no LD source)
    for m in ld_models:                                    # SECONDARY: MS -> HKCD (different source)
        specs.append(("urb3dcd_v2_ms", "hkcd", m, False))
    for m in ld_models:                                    # SECONDARY: LD -> MS (different target)
        specs.append(("urb3dcd_v2_ld", "urb3dcd_v2_ms", m, False))

    rows = []
    # Cache target-side loads/refs/oracle per (target, model) so we read each npz set once.
    test_cache = {}
    refs_cache = {}
    oracle_cache = {}

    for source, target, model, is_primary in specs:
        # Source (c, lambda) - the thing we transfer.
        src_params = load_two_moment_params(source)
        tgt_params = load_two_moment_params(target)
        if model not in src_params or model not in tgt_params:
            continue                                       # model not present in a dataset (e.g. randla on LD)

        # Target test predictions + per-cube stats (computed from the TARGET predictions).
        if (target, model) not in test_cache:
            test_cache[(target, model)] = load_target_test(target, model)
        test_recs, stats_by_scene = test_cache[(target, model)]

        # Target references (fixed_05 / f1_optimal / two_moment refit) straight from the main table.
        if target not in refs_cache:
            refs_cache[target] = load_main_table_refs(target)
            oracle_cache[target] = load_diag(target)[0]    # {model: per-cube oracle ceiling (test mean per-scene)}
        refs = refs_cache[target]
        oracle = oracle_cache[target]

        c_src, lam_src, fb_src = src_params[model]["c"], src_params[model]["lambda"], src_params[model]["fallback_tau"]
        c_tgt, lam_tgt, fb_tgt = tgt_params[model]["c"], tgt_params[model]["lambda"], tgt_params[model]["fallback_tau"]

        # (a) transferred: SOURCE (c, lambda). Fallback tau = TARGET's f1_optimal tau, so this arm
        # differs from the refit arm ONLY in (c, lambda) -> the gap isolates the (c, lambda) change.
        transferred_f1 = mean_test_f1(test_recs, stats_by_scene, c_src, lam_src, fb_tgt)
        # Robustness variant: also carry the SOURCE's fallback tau (zero target-side quantities at
        # all). The fallback only fires on cubes with < MIN_NC no-change points, so the two should
        # be within rounding; we report it to show the transferred number is not a fallback artifact.
        transferred_f1_srcfb = mean_test_f1(test_recs, stats_by_scene, c_src, lam_src, fb_src)

        # (b) refit: TARGET (c, lambda). Recomputed here and cross-checked against the main table.
        refit_f1 = mean_test_f1(test_recs, stats_by_scene, c_tgt, lam_tgt, fb_tgt)
        refit_ref = refs.get((model, "two_moment"))
        if refit_ref is not None and abs(refit_f1 - refit_ref) > 5e-3:
            print("  WARN recomputed refit F1 %.4f != main_table two_moment %.4f (%s/%s)" % (
                refit_f1, refit_ref, target, model))

        fixed_05_f1 = refs.get((model, "fixed_05"), float("nan"))
        f1_optimal_f1 = refs.get((model, "f1_optimal"), float("nan"))
        oracle_f1 = oracle.get(model, float("nan"))

        # Validity: both ends must be healthy for the transfer to MEAN anything.
        if model in FAILED.get(source, set()):
            status = "na_source_failed"
        elif model in FAILED.get(target, set()):
            status = "na_target_failed"
        else:
            status = "valid"

        gap = refit_f1 - transferred_f1                    # how much we lose by NOT refitting (small = transfers)
        transferred_minus_f1opt = transferred_f1 - f1_optimal_f1   # >=0 => no retuning needed
        headroom = oracle_f1 - f1_optimal_f1               # per-cube headroom over the tuned global threshold
        frac_kept = (transferred_minus_f1opt / headroom) if headroom > 1e-9 else float("nan")

        rows.append({
            "source": source, "target": target, "model": model,
            "status": status, "is_primary": int(is_primary),
            "src_c": r(c_src), "src_lambda": r(lam_src),
            "transferred_f1": r(transferred_f1), "transferred_f1_srcfb": r(transferred_f1_srcfb),
            "tgt_c": r(c_tgt), "tgt_lambda": r(lam_tgt),
            "refit_f1": r(refit_f1), "gap": r(gap),
            "fixed_05_f1": r(fixed_05_f1), "f1_optimal_f1": r(f1_optimal_f1),
            "oracle_f1": r(oracle_f1),
            "transferred_minus_f1opt": r(transferred_minus_f1opt),
            "headroom": r(headroom), "frac_headroom_kept": r(frac_kept),
        })

    # Write the table.
    os.makedirs("results", exist_ok=True)
    fields = ["source", "target", "model", "status", "is_primary",
              "src_c", "src_lambda", "transferred_f1", "transferred_f1_srcfb",
              "tgt_c", "tgt_lambda", "refit_f1", "gap",
              "fixed_05_f1", "f1_optimal_f1", "oracle_f1",
              "transferred_minus_f1opt", "headroom", "frac_headroom_kept"]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print("wrote %s (%d rows)" % (args.out, len(rows)))

    # ---- Console summaries ----
    def block(title, predicate):
        sel = [r_ for r_ in rows if predicate(r_)]
        if not sel:
            return
        print("\n== %s ==" % title)
        print("  %-26s %-7s %8s %8s %8s   %8s %8s %8s   %8s" % (
            "source->target/model", "status", "transf", "refit", "gap",
            "fixed05", "f1opt", "oracle", "tr-f1opt"))
        for r_ in sel:
            tag = "%s->%s/%s" % (r_["source"].replace("urb3dcd_v2_", ""),
                                 r_["target"].replace("urb3dcd_v2_", ""), r_["model"])
            print("  %-26s %-7s %8.4f %8.4f %+8.4f   %8.4f %8.4f %8.4f   %+8.4f" % (
                tag[:26], r_["status"].replace("na_", "")[:7],
                r_["transferred_f1"], r_["refit_f1"], r_["gap"],
                r_["fixed_05_f1"], r_["f1_optimal_f1"], r_["oracle_f1"],
                r_["transferred_minus_f1opt"]))

    block("PRIMARY (sim->real into HKCD) - the Section 0 claim", lambda r_: r_["is_primary"] == 1)
    block("SECONDARY MS->HKCD", lambda r_: r_["is_primary"] == 0 and r_["target"] == "hkcd")
    block("SECONDARY LD->MS", lambda r_: r_["is_primary"] == 0 and r_["target"] == "urb3dcd_v2_ms")

    # ---- Verdict on the valid PRIMARY rows (the sim->real claim) ----
    primary_valid = [r_ for r_ in rows if r_["is_primary"] == 1 and r_["status"] == "valid"]
    if primary_valid:
        gaps = [r_["gap"] for r_ in primary_valid]
        deltas = [r_["transferred_minus_f1opt"] for r_ in primary_valid]
        print("\n== VERDICT (valid PRIMARY rows, n=%d) ==" % len(primary_valid))
        print("  mean transfer gap (refit - transferred) = %+.4f   (max |gap| = %.4f)" % (
            float(np.mean(gaps)), float(np.max(np.abs(gaps)))))
        print("  mean (transferred - f1_optimal)         = %+.4f   (>=0 on %d/%d models)" % (
            float(np.mean(deltas)), int(np.sum(np.array(deltas) >= -1e-9)), len(deltas)))
        print("  transferred >= f1_optimal means: no per-dataset threshold tuning was needed.")


if __name__ == "__main__":
    main()
