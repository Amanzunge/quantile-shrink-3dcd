# Mathematical and theoretical account of the per-cube threshold method

This is the derivation behind Section 0 of CLAUDE.md: why the per-cube threshold
exists at all, why the two-moment form is its Gaussian special case, why the
empirical-quantile form dominates it, and what the shrinkage weight does. It is
written to be lifted into the paper's Method section.

## 1. Setup and notation

A scene is tiled into cubes C_1..C_K. The Siamese model assigns every point p of
the second epoch a change score s_p = P(change | p) in [0,1] (softmax of the
NN-propagated logits). A calibration method is a rule producing per-cube
thresholds tau_1..tau_K; the decision is

    y_hat_p = 1{ s_p > tau_i }   for p in C_i,

and the target metric is the F1 of the change class over the scene. The only
information available at test time is the scores themselves; in particular the
PREDICTED no-change subset of cube i is

    N_i = { p in C_i : s_p <= 1/2 },     n_i = |N_i|,

which is a label-free, contaminated, TRUNCATED sample of the true no-change
score distribution (contaminated by false negatives, truncated at 1/2 by
construction).

## 2. Why a per-cube threshold: heterogeneous local false-alarm rates

Let F_i be the score distribution of TRUE no-change points in cube i. Under a
threshold tau the local false-positive rate is

    FPR_i(tau) = 1 - F_i(tau).

The cubes are heterogeneous (vegetation, density mismatch between epochs,
registration residue), so the F_i differ, and one GLOBAL tau implies wildly
different FPR_i across cubes: too permissive exactly where the model runs hot.
The natural repair is to equalize the local false-alarm level: choose

    tau_i = F_i^{-1}(1 - alpha)                                   (2.1)

for a single global exceedance level alpha. This is the Neyman-Pearson-style
reading of per-cube calibration: one global operating point (alpha), local
thresholds that realize it. The single global level is then tuned on validation
data to the F1-optimal operating point, which is what the grid search over the
method's scalars does.

## 3. two_moment is the Gaussian plug-in estimator of (2.1)

If F_i were Gaussian with mean mu_i and standard deviation sigma_i, then

    F_i^{-1}(1 - alpha) = mu_i + z_{1-alpha} * sigma_i,

with z the standard normal quantile. Substituting the sample moments of N_i and
absorbing the truncation/contamination bias into one global offset c gives
exactly the two-moment rule

    tau_i = clip( c + mean(N_i) + lambda * std(N_i) ),   lambda ~ z_{1-alpha}.

So two_moment is the plug-in of (2.1) under three assumptions, each of which
fails in practice:

  (a) SHAPE. Scores are bounded in [0,1], skewed and often bimodal; the Gaussian
      tail formula mislocates the high quantile. Worse, ONE global lambda must
      fit all cubes simultaneously, but the mapping from (mu_i, sigma_i) to the
      true tail location varies per cube; the affine family in the two moments
      is not expressive enough. (Empirically: Spearman correlation between the
      fitted two_moment taus and the per-cube oracle taus is ~0 on the primary
      benchmark - the per-cube variation it produces is noise.)
  (b) TRUNCATION. N_i is cut at 1/2, so mean(N_i), std(N_i) estimate moments of
      the truncated law; the bias depends on the shape of F_i near 1/2 and is
      NOT a constant across cubes, so c cannot correct it uniformly.
  (c) CONTAMINATION. Missed changes (s <= 1/2) sit in the upper tail of N_i and
      pull mean and std upward with unbounded leverage (std is quadratic in
      outliers), precisely in change-dense cubes.

## 4. The empirical (conformal) quantile estimator

Replace the parametric plug-in with the order statistic itself:

    q_hat_alpha(N_i) = s_(r) ,   r = ceil( alpha * (n_i + 1) ),      (4.1)

the conformal choice of rank. The method's threshold is

    tau_i = clip( c + q_hat_alpha(N_i) ).

Finite-sample validity (the conformal/Mondrian argument): if the scores in N_i
are exchangeable draws from a common (truncated) law and p is a further draw
from it, then by exchangeability of the n_i + 1 values

    P( s_p > q_hat_alpha(N_i) ) <= 1 - alpha,                        (4.2)

with NO assumption on the shape of the distribution. Applied per cube, (4.2) is
group-conditional (Mondrian) false-alarm control with the cubes as groups: the
quantity two_moment tried to construct through a Gaussian model is obtained
distribution-free. The offset c plays the same role as before - it carries the
threshold from the truncated law's tail (which cannot exceed 1/2) up to the
decision boundary - but it no longer has to repair a shape error, only the
truncation gap, which is far closer to constant across cubes.

Robustness: an alpha-quantile ignores everything above rank r, so contamination
of mass eps < 1 - alpha in the upper tail shifts it by at most the local
inter-order-statistic spacing; the moments have no such breakdown protection.

## 5. Shrinkage: the small-cube variance repair

The asymptotic variance of an empirical quantile is

    Var( q_hat_alpha ) ~ alpha (1 - alpha) / ( n_i * f_i( q_alpha )^2 ),

so for small n_i the per-cube estimate is noise - the diagnosed IndoorCD
failure (median 7 cubes/scene, 78% zero-change cubes, tiny N_i). Model the cube
quantiles hierarchically: q_i scattered around the scene-level quantile q_bar
(random-effects view). The linear-blend estimator

    q_tilde_i = w_i * q_hat_alpha(N_i) + (1 - w_i) * q_hat_alpha(N_scene),
    w_i = n_i / (n_i + k),                                           (5.1)

is the standard empirical-Bayes shrinkage form: exact posterior mean in the
Gaussian random-effects case with k the ratio of within-cube noise variance to
between-cube signal variance, and a well-behaved working estimator in general.
The mean-squared-error tradeoff: shrinkage multiplies the estimation variance
by w_i^2 at the price of a bias toward q_bar of size (1-w_i)*|q_i - q_bar|;
for n_i << k the variance reduction dominates. Two limits anchor the family:

    k = 0        -> pure per-cube quantile (w_i = 1 whenever n_i > 0);
    k -> infinity -> tau_i = c + q_bar, a GLOBAL threshold.

The second limit means the hypothesis family CONTAINS the tuned global
threshold as a special case. This is why quantile_shrink cannot lose to
f1_optimal by more than grid resolution plus val-to-test generalization error -
and why it replaces the hard MIN_NC fallback: the fallback is now the smooth
k -> infinity end of the same family rather than a discontinuous switch.

## 6. Fitting and generalization

(c, alpha, k) are chosen by empirical risk maximization over a finite grid
(61 x 14 x 7 = 5,978 candidates), objective = mean per-scene validation F1,
identical protocol to the two_moment fit. The scene F1 as a function of the
per-cube threshold vector is piecewise constant, so grid search over order
statistics is exact - there is no continuous-optimization failure mode. With a
finite class of a few thousand candidates evaluated on thousands of cubes,
uniform convergence is mild; empirically the val-selected parameters carry to
test (the val and test improvements appear in the same combinations).

## 7. Why NOT a richer estimator: the information bound

The estimator autopsy regresses the per-cube ORACLE threshold on every
label-free cube statistic available (n_i, moments, median/MAD, tail quantiles
of N_i, change-side quantiles, population sizes): 5-fold cross-validated R^2
tops out around 0.2 on the primary benchmark (0.4 at the indoor scope floor),
for linear models and random forests alike. That number upper-bounds the
oracle-tracking ability of ANY calibrator built from these statistics -
including this one. The design conclusion: spend the parameter budget on a
correct, low-variance estimate of the ONE robust local functional (the tail
quantile) rather than on a richer map from cube statistics to tau, which would
chase unpredictable variation (empirically: the two_moment_shrink variant
drives its shrinkage weight to the global limit - the fit itself asks to delete
the extra local structure).

Split-half validation of the target: fitting the per-cube oracle on one random
half of each cube and scoring it on the other half retains 94-98% of the
oracle headroom on MS, 66-99% on HKCD, 84-100% on IndoorCD (48-73% on LD), so
the headroom quantity being chased is real signal, not oracle overfitting; the
un-captured remainder requires label-side or spatial-context information, which
no per-cube score statistic can supply.

## 8. Interpretation of the fitted values

  alpha in [0.97, 1.0] on healthy models: the threshold sits just above the
      local no-change mass, as (2.1) prescribes.
  c > 0 : the truncation gap between the (<= 1/2) sample tail and the actual
      decision boundary, plus the global operating-point tuning.
  k in [64, 1024] : cubes with fewer than roughly a hundred to a thousand
      predicted-no-change points lean substantially on the scene quantile -
      the soft version of the old MIN_NC = 10 fallback, engaged earlier and
      proportionally.
  Degenerate fits (alpha = 0.5, c < 0) occur exactly on models that failed the
      AUC gate (HKCD kpconv): with no usable score separation the calibration
      family has nothing to calibrate, matching the two_moment behavior.
