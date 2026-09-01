# Task 12 synthesis (final benchmark + ablations)

Status: COMPLETE. All 66 deep GPU ablation runs finished 2026-07-12 (Kaggle +
Colab; see `ablations/RUN_PLAN.md`). On 2026-07-14 the ablation tables and
figures were REBUILT around quantile_shrink after the paper layer went
two_moment-free; this document was rewritten the same day to match.

HISTORY NOTE (2026-07-14). The original version of this file (2026-06-24)
narrated the ablations in terms of two_moment, then the headline method. On
2026-07-13 the shrunk per-cube quantile (quantile_shrink) superseded two_moment
(see `results/METHOD_V2_SYNTHESIS.md` for the full comparison and why it won);
on 2026-07-14 the researcher removed two_moment from all paper-facing tables
and figures. two_moment ablation numbers are regenerable from predictions/ via
`src/calibrate.py` if ever needed.

## 1. Final synthesis tables (pre-existing, FROZEN, unchanged)
- `results/main_table.csv`        132 rows (LD 30 + MS 36 + HKCD 36 + IndoorCD 30).
- `results/transfer_table.csv`    16 rows (Task 10).
- `results/scope_limit_table.csv` 19 rows (Task 11).
Task 12 adds ablation outputs ONLY; predictions/ and main_table.csv are untouched.

## 2. Task 12 artefacts (rebuilt 2026-07-14, two_moment-free)
- `results/ablation_cube_geometry.csv`  model-free boundary, all 4 datasets x 3 cube sizes.
- `results/ablation_cube.csv`           66 rows, cube-size sweep (22 main anchors reused).
- `results/ablation_fps.csv`            37 rows, FPS sweep, LD+MS only (9 main anchors reused).
  Calibration columns: `quantile_shrink_test`, `quantile_shrink_gain` (vs the
  per-run tuned global `f1_optimal_test`), `oracle_ceiling`, `oracle_headroom`,
  `mean_shrink_w_test`, fitted `c`/`alpha`/`k`, `f1_optimal_tau`, plus the
  geometry stats (cubes/scene, drop %, zero-change %).
- Figures (`src/plot_ablations.py`, six, paper style):
  `fig_cube_geometry_boundary.png`   model-free: zero-change % and cubes/scene vs cube size
  `fig_cube_gain_headroom.png`       per dataset: oracle headroom vs realised quantile_shrink gain
  `fig_fps_sweep.png`                LD/MS: gain and headroom vs FPS size
  `fig_gain_vs_headroom_scatter.png` every run as one point: headroom vs realised gain
  `fig_cube_auc_by_model.png`        pooled test AUC vs cube extent (the H1 gate)
  `fig_cube_absolute_f1.png`         absolute F1 (global vs per-cube vs oracle) vs cube extent
- Code: `src/build_ablation_tables.py` (fits quantile_shrink per run via the
  `src/calibrate_v2.py` machinery, same val-grid protocol as the main table);
  `--fps-native`/`--cache-root` on train/predict (+ predict_icp `--cache-root`,
  inspect_dataset `--cube-size`); Kaggle/Colab runner scripts live on the cloud
  side (`ablations/RUN_PLAN.md` documents the run matrix).

## 3. The model-free scope-limit boundary (no GPU needed)

Smaller cubes -> more cubes/scene, higher min_points drop rate, and more zero-change cubes.
All three signals move the outdoor datasets toward the IndoorCD room-scale floor:

| dataset  | cube | cubes/scene | drop% | zero-change% |
|----------|------|-------------|-------|--------------|
| LD       | 25 m | 411         | 25.5  | 16.9 |
| LD       | 50 m | 140         | 9.8   | 2.9  |
| LD       | 75 m | 70          | 8.8   | 1.4  |
| MS       | 25 m | 452         | 19.3  | 15.4 |
| MS       | 50 m | 142         | 9.4   | 3.0  |
| MS       | 75 m | 70          | 8.5   | 1.4  |
| HKCD     | 25 m | 703         | 3.8   | 38.6 |
| HKCD     | 50 m | 180         | 0.3   | 14.6 |
| HKCD     | 75 m | 80          | 0.4   | 4.5  |
| IndoorCD | 1.0 m| 11          | 26.1  | 83.5 |
| IndoorCD | 1.5 m| 7           | 23.9  | 78.4 |
| IndoorCD | 2.0 m| 4           | 24.0  | 74.3 |

IndoorCD is the LIMITING CASE of one continuum, not a separate regime: even its largest
2.0 m cube has 74% zero-change cubes and 4 cubes/scene, beyond the smallest outdoor cube.

## 4. Headroom vs realised gain (quantile_shrink, from ablation_cube.csv)

Per-cube oracle headroom (signal that EXISTS) grows as cubes shrink; the realised
quantile_shrink gain does not keep pace. ICP rows (head / gain vs tuned global tau):

| dataset | model | 25 m head/gain | 50 m head/gain | 75 m head/gain |
|---------|-------|----------------|----------------|----------------|
| LD   | icp | +0.087 / +0.000 | +0.048 / -0.000 | +0.037 / -0.000 |
| MS   | icp | +0.024 / -0.001 | +0.010 / -0.001 | +0.021 / +0.001 |
| HKCD | icp | +0.109 / +0.055 | +0.040 / +0.043 | +0.022 / +0.044 |

HKCD ICP is the visible improvement over the two_moment era: where two_moment
LOST badly (down to -0.232 at 75 m), quantile_shrink now gains +0.04..+0.05 at
every cube size (the truncated non-Gaussian no-change distribution that broke
the moment estimator; see METHOD_V2_SYNTHESIS.md Section 2).

Deep-model example of the same widening gap (LD siamese_pointnet, 75/50/25 m):
headroom 0.037 / 0.056 / 0.111 vs realised gain -0.000 / +0.005 / +0.031 --
gain grows, but captures a shrinking fraction of the growing headroom.

At the IndoorCD scope floor the headroom is the LARGEST in the project (deep
mean ~+0.09..+0.14 per cube size) yet the realised gain is ~0 on all 15 rows
(range -0.011..+0.015). Unlike two_moment (gain <= 0 on ALL 5 models at 1.5 m),
quantile_shrink no longer systematically loses there -- but it realises
essentially none of the signal either.

## 5. Section 5.3 hypotheses (formed FROM the results; all four CONFIRMED on the full deep data)

- **H1 (continuous scope limit -- CONFIRMED, model-free + model gate).** Shrinking the cube
  monotonically raises cubes/scene, the drop rate, and the zero-change-cube fraction (Section 3),
  and the models themselves fail the AUC gate at 25 m (MS kpconv AUC 0.50, MS pointnet 0.57,
  MS siamgcn 0.59; HKCD kpconv 0.59 -- see fig_cube_auc_by_model.png). The room-scale collapse
  is the limit of this curve, not a discontinuity.
- **H2 (per-cube headroom grows as cubes shrink -- CONFIRMED).** The per-cube oracle ceiling
  over a tuned global tau increases as cubes shrink within every dataset (e.g. LD pointnet
  75/50/25 m = 0.037/0.056/0.111; deep means LD 0.037->0.095, HKCD 0.063->0.145) and is largest
  at the IndoorCD floor. The split-half autopsy (METHOD_V2_SYNTHESIS.md Section 3) shows this
  headroom is real, not small-sample oracle optimism.
- **H3 (estimators capture little of it at small scale -- CONFIRMED, restated for
  quantile_shrink).** The headline estimator beats the tuned global tau on 61% of the 103
  ablation runs (63/103; two_moment was flat/negative), so the sign of the story improved --
  but the MEDIAN captured fraction of the oracle headroom is ~1% (1.1% on both sweeps), and
  the oracle-vs-realised gap still WIDENS as cubes shrink (fig_gain_vs_headroom_scatter.png).
  "Signal exists, label-free estimators realise little of it at small scale" stands; the
  autopsy R^2 bound (oracle tau only weakly predictable from any per-cube score statistic)
  says the residual gap is information-theoretic, not a fixable estimator deficiency.
- **H4 (FPS is orthogonal to the per-cube story -- CONFIRMED).** FPS changes only the model
  input resolution, not the cube partition. quantile_shrink gain is roughly FPS-invariant
  (LD +0.001..+0.003 across 256..1024; MS shows no monotone trend), while model quality moves
  with FPS through a DENSITY effect, not a calibration one: RandLA is robust, but PointNet and
  SiamGCN degrade at high FPS on MS (mean pooled AUC 0.90 at 512 vs 0.81 at 4096). Calibration
  gain tracks model quality, not input resolution.

## 6. Honest one-paragraph project synthesis

Across all four datasets the per-cube threshold story is now precisely bounded. A real,
split-half-validated per-cube ORACLE headroom exists over a per-dataset tuned global
threshold, and it is LARGEST exactly where cubes are smallest and most heterogeneous
(room scale). The headline quantile_shrink calibrator beats the tuned global threshold on
61% of the 103 ablation runs -- including runs where the retired two_moment estimator lost
outright (HKCD ICP at every cube size) -- and its biggest wins land on the more-miscalibrated
models (MS siamgcn 75 m +0.123, MS siamgcn fps-1024 +0.058). But the median captured fraction
of the oracle headroom is ~1%, its clear losses concentrate on models that failed training
(MS pointnet 25 m -0.101 at AUC 0.57; HKCD kpconv -0.031 at AUC 0.57), and at the IndoorCD
floor it realises essentially none of the largest headroom in the project. The contribution
is therefore bounded and honest: the per-cube signal is real and quantified, a distribution-
free shrunk quantile extracts the extractable part where two moments could not, and the
remaining oracle gap at small scale is information-theoretic (the oracle threshold is only
weakly predictable from any label-free per-cube statistic), which points future work at
label-side or spatial-context information rather than at yet another score-statistic estimator.
