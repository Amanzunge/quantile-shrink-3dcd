"""
plot_method_concept.py - Paper Fig. 2: the idea behind quantile-shrink, drawn from analytic
(logit-normal) score distributions with symbolic labels only, so it depends on no result file.
  (a) two cubes with different no-change score distributions: one shared threshold tau gives
      different local false-positive rates, one shared quantile level alpha gives the same rate;
  (b) the rule for one cube in four steps: predicted no-change sample N_i, sort and read the
      quantile at the conformal rank, shrink toward the scene quantile, add the offset c.
Torch-free, local CPU:  python src/plot_method_concept.py
Writes results/paper_figures/fig_method_concept.png (600 dpi).
"""
import os
import math

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

OUT_DIR = os.path.join("results", "paper_figures")
TEAL, CORAL, PURPLE, PURPLE_L, GRAY, GRAY_L, INK = "#1D9E75", "#D85A30", "#534AB7", "#AFA9EC", "#888780", "#D3D1C7", "#2C2C2A"
plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 7.5,
                     "axes.titlesize": 8, "xtick.labelsize": 7.5, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.linewidth": 0.8})
T = np.linspace(0.0015, 0.9985, 1200)          # score grid used by every curve

# ---- analytic logit-normal helpers: score s = sigmoid(mu + sd * Z), Z standard normal ----
_erf = np.vectorize(math.erf)


def Phi(z):
    return 0.5 * (1.0 + _erf(np.asarray(z) / math.sqrt(2.0)))


def Phi_inv(a):
    lo, hi = -10.0, 10.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if float(Phi(mid)) < a:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def logit(t):
    return np.log(t / (1.0 - t))


def sigmoid(v):
    return 1.0 / (1.0 + np.exp(-v))


def dens(t, mu, sd):
    z = (logit(t) - mu) / sd
    return np.exp(-0.5 * z * z) / math.sqrt(2 * math.pi) / (sd * t * (1.0 - t))


def cdf(t, mu, sd):
    return Phi((logit(t) - mu) / sd)


def qf(a, mu, sd):
    return float(sigmoid(mu + sd * Phi_inv(a)))


def truncated_inverse(G, a):
    """Inverse of a CDF G on [0, 1/2] by bisection."""
    lo, hi = 0.0005, 0.5
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if G(mid) < a:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def score_axis(ax, extra=None, show_half=True):
    ticks, labels = [0.0, 1.0], ["0", "1"]
    if show_half:
        ticks.append(0.5)
        labels.append(r"$\frac{1}{2}$")
    for value, label in (extra or []):
        ticks.append(value)
        labels.append(label)
    order = np.argsort(ticks)
    ax.set_xticks([ticks[i] for i in order])
    ax.set_xticklabels([labels[i] for i in order])
    ax.set_xlim(0, 1)


def tail_panel(ax, mu, sd, tau, label_tail, title, tick_label):
    f = dens(T, mu, sd)
    left, right = T <= tau, T >= tau
    ax.fill_between(T[left], f[left], color=TEAL, alpha=0.85, lw=0)
    ax.fill_between(T[right], f[right], color=CORAL, alpha=0.9, lw=0)
    ax.plot(T, f, color=INK, lw=0.6)
    ax.axvline(tau, color=PURPLE, lw=1.4)
    ax.set_yticks([])
    ax.set_ylim(0, f.max() * 1.18)
    ax.set_title(title, fontsize=8)
    xt = min(tau + 0.16, 0.74)
    ax.annotate(label_tail, xy=(min(tau + 0.05, 0.97), dens(min(tau + 0.05, 0.97), mu, sd) * 0.4),
                xytext=(xt, f.max() * 0.78), fontsize=7.5, color=CORAL,
                arrowprops=dict(arrowstyle="-", color=CORAL, lw=0.6))
    score_axis(ax, [(tau, tick_label)], show_half=False)
    ax.set_xlabel(r"score $s_p$", labelpad=1)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    fig, axes = plt.subplots(2, 4, figsize=(6.3, 3.75),
                             gridspec_kw={"hspace": 1.05, "wspace": 0.16, "left": 0.02, "right": 0.99,
                                          "top": 0.86, "bottom": 0.11})
    # ---- row (a): shared tau versus shared alpha, two cubes ------------------------------
    cube1, cube2, alpha_fig, tau = (-3.0, 1.0), (-0.8, 1.1), 0.85, 0.33   # alpha only fixes the picture
    t1, t2 = qf(alpha_fig, *cube1), qf(alpha_fig, *cube2)
    tail_panel(axes[0, 0], *cube1, tau, r"$\mathrm{FPR}_1(\tau)$", r"cube $C_1$, shared $\tau$", r"$\tau$")
    tail_panel(axes[0, 1], *cube2, tau, r"$\mathrm{FPR}_2(\tau)$", r"cube $C_2$, shared $\tau$", r"$\tau$")
    tail_panel(axes[0, 2], *cube1, t1, r"$1-\alpha$", r"cube $C_1$, shared $\alpha$", r"$\tau^*_1$")
    tail_panel(axes[0, 3], *cube2, t2, r"$1-\alpha$", r"cube $C_2$, shared $\alpha$", r"$\tau^*_2$")

    # ---- row (b): the rule for one cube, four steps ---------------------------------------
    mu0, sd0 = -1.0, 1.0                  # truly unchanged points of a cube where the model runs hot
    mu1, sd1, w1 = 1.5, 1.1, 0.35         # changed points
    f0, f1 = dens(T, mu0, sd0), w1 * dens(T, mu1, sd1)
    den = float(cdf(0.5, mu0, sd0)) + w1 * float(cdf(0.5, mu1, sd1))

    def G(t):
        return (float(cdf(t, mu0, sd0)) + w1 * float(cdf(t, mu1, sd1))) / den

    n = 14
    sample = [truncated_inverse(G, k / (n + 1)) for k in range(1, n + 1)]
    r = math.ceil(0.85 * (n + 1))
    q_cube, q_scene, w, c = sample[r - 1], 0.22, 0.45, 0.22
    q_mix = w * q_cube + (1 - w) * q_scene
    tau_i = q_mix + c
    top = max(f0.max(), f1.max())

    ax = axes[1, 0]
    ax.axvspan(0, 0.5, color=GRAY_L, alpha=0.55, lw=0)
    ax.fill_between(T, f0, color=TEAL, alpha=0.5, lw=0)
    below = T <= 0.5
    ax.fill_between(T[below], f1[below], color=CORAL, alpha=0.75, lw=0)
    ax.plot(T, f0, color=TEAL, lw=0.9)
    ax.plot(T, f1, color=CORAL, lw=0.9)
    ax.set_yticks([])
    ax.set_ylim(0, top * 1.15)
    score_axis(ax)
    ax.set_title(r"1. scores of cube $i$; shaded: $N_i$", fontsize=8)

    ax = axes[1, 1]
    ax.hlines(0, 0, 0.5, color=GRAY_L, lw=1)
    for k, value in enumerate(sample, start=1):
        ax.plot([value], [0], "o", color=PURPLE if k == r else TEAL, ms=6.5 if k == r else 4)
    ax.annotate(r"rank $r=\lceil\alpha(n_i+1)\rceil$", xy=(q_cube, 0.05), xytext=(q_cube - 0.33, 0.6),
                fontsize=7.5, color=PURPLE, arrowprops=dict(arrowstyle="-", color=PURPLE, lw=0.6))
    ax.set_ylim(-0.4, 1.0)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlim(0, 0.5)
    ax.set_xticks([0, q_cube, 0.5])
    ax.set_xticklabels(["0", r"$q_\alpha(N_i)$", r"$\frac{1}{2}$"])
    ax.set_title(r"2. sort $N_i$; quantile at rank $r$", fontsize=8)

    ax = axes[1, 2]
    ax.hlines(0, 0, 0.5, color=GRAY_L, lw=1)
    ax.plot([q_scene, q_scene], [-0.22, 0.9], color=GRAY, lw=1.2, ls="--")
    ax.plot([q_cube], [0], "o", color=TEAL, ms=6.5, zorder=3)
    ax.plot([q_mix], [0], "o", color=PURPLE, ms=8.5, mfc="none", mew=1.6, zorder=4)
    ax.annotate("", xy=(q_mix, 0.0), xytext=(q_cube, 0.0),
                arrowprops=dict(arrowstyle="-|>", color=PURPLE, lw=0.8, shrinkA=4, shrinkB=5))
    ax.text(q_cube, 0.33, r"$q_\alpha(N_i)$", ha="center", fontsize=7.5, color=TEAL)
    ax.text(0.49, -0.47, r"$w_i\,q_\alpha(N_i)+(1-w_i)\,q_\alpha(N_{\mathrm{scene}})$", ha="right", fontsize=7.4, color=PURPLE)
    ax.text(q_scene - 0.01, 0.62, r"$q_\alpha(N_{\mathrm{scene}})$", ha="right", fontsize=7.5, color=GRAY)
    ax.set_ylim(-0.7, 0.9)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlim(0, 0.5)
    ax.set_xticks([0, 0.5])
    ax.set_xticklabels(["0", r"$\frac{1}{2}$"])
    ax.set_title(r"3. shrink toward the scene quantile", fontsize=8)

    ax = axes[1, 3]
    ax.fill_between(T, f0, color=TEAL, alpha=0.5, lw=0)
    ax.fill_between(T, f1, color=CORAL, alpha=0.35, lw=0)
    ax.plot(T, f0, color=TEAL, lw=0.9)
    ax.plot(T, f1, color=CORAL, lw=0.9)
    ax.axvspan(tau_i, 1, color=PURPLE_L, alpha=0.35, lw=0)
    ax.axvline(q_mix, color=PURPLE, lw=1.0, ls="--")
    ax.axvline(tau_i, color=PURPLE, lw=1.6)
    ax.annotate("", xy=(tau_i, top * 1.0), xytext=(q_mix, top * 1.0),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.8))
    ax.text((q_mix + tau_i) / 2, top * 1.04, r"$c$", ha="center", va="bottom", fontsize=8)
    ax.text(tau_i + 0.02, top * 0.55, "change", color=PURPLE, fontsize=7.5)
    ax.set_yticks([])
    ax.set_ylim(0, top * 1.2)
    score_axis(ax, [(tau_i, r"$\tau_i$")], show_half=False)
    ax.set_title(r"4. offset $c$, threshold $\tau_i$, decision", fontsize=8)

    fig.text(0.005, 0.965, "(a)", ha="left", va="center", fontsize=9, fontweight="bold")
    fig.text(0.005, 0.455, "(b)", ha="left", va="center", fontsize=9, fontweight="bold")
    fig.legend(handles=[Patch(color=TEAL, alpha=0.85, label="truly unchanged points"),
                        Patch(color=CORAL, alpha=0.9, label="changed points / above threshold"),
                        Patch(color=GRAY_L, alpha=0.8, label=r"visible sample $N_i$ ($s_p \leq 1/2$, no labels)")],
               loc="lower center", ncol=3, frameon=False, fontsize=7.8, bbox_to_anchor=(0.5, -0.045),
               handlelength=1.2, columnspacing=1.5)
    out = os.path.join(OUT_DIR, "fig_method_concept.png")
    fig.savefig(out, dpi=600, bbox_inches="tight", pad_inches=0.03)
    print("wrote", out)


if __name__ == "__main__":
    main()
