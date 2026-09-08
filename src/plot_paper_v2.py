"""
plot_paper_v2.py - Draft-v3 paper figures, written to paper/figures/.

Replaces the critical-difference diagram (too much statistical machinery for the
message) with two plain-language figures, and adds a figure for the R^2
predictability bound, which until now was a text-only claim:

  fig_head_to_head.png   (a) win/tie/loss record of quantile-shrink against each
                         baseline over the 22 combos; (b) mean rank per method.
  fig_forest.png         per-combination F1 difference vs the tuned global
                         threshold with paired-bootstrap 95% CIs (visual Table 3).
  fig_signal_bound.png   (a) split-half check that the oracle headroom is real;
                         (b) cross-validated R^2 of the oracle threshold from
                         label-free cube statistics (the reachability bound).

Torch-free, local CPU:  python src/plot_paper_v2.py
"""

import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import rankdata

# Same palette/ink as the other paper figures.
M_COLOR = {"quantile_shrink": "#2a78d6", "f1_optimal": "#1baf7a", "temperature_scaling": "#eda100",
           "platt": "#4a3aa7", "isotonic": "#e87ba4", "fixed_05": "#8a8985", "otsu_cube": "#b0552a"}
M_LABEL = {"quantile_shrink": "quantile-shrink (ours)", "f1_optimal": "tuned global tau",
           "temperature_scaling": "temperature", "platt": "Platt", "isotonic": "isotonic",
           "fixed_05": "fixed 0.5", "otsu_cube": "per-cube Otsu"}
INK, SOFT, GRID = "#0b0b0b", "#52514e", "#eceae6"
WIN, TIE, LOSS = "#2a78d6", "#d8d7d3", "#c9366b"

METHODS7 = ["fixed_05", "platt", "isotonic", "temperature_scaling", "f1_optimal",
            "otsu_cube", "quantile_shrink"]
BASELINES = ["fixed_05", "otsu_cube", "platt", "isotonic", "temperature_scaling", "f1_optimal"]
DS_ORDER = ["urb3dcd_v2_ld", "urb3dcd_v2_ms", "hkcd", "indoorcd"]
DS_SHORT = {"urb3dcd_v2_ld": "LD", "urb3dcd_v2_ms": "MS", "hkcd": "HKCD", "indoorcd": "Indoor"}
DS_LABEL = {"urb3dcd_v2_ld": "Urb3DCD-V2 LD", "urb3dcd_v2_ms": "Urb3DCD-V2 MS",
            "hkcd": "HKCD", "indoorcd": "IndoorCD"}
DS_COLOR = {"urb3dcd_v2_ld": "#2a78d6", "urb3dcd_v2_ms": "#1baf7a",
            "hkcd": "#eda100", "indoorcd": "#e87ba4"}
MODEL_ORDER = ["icp_euclidean", "siamese_pointnet", "siamese_pointnet2",
               "siamese_kpconv", "randla", "siamgcn"]
MODEL_LABEL = {"icp_euclidean": "ICP", "siamese_pointnet": "PointNet",
               "siamese_pointnet2": "PointNet++", "siamese_kpconv": "KPConv",
               "randla": "RandLA-Net", "siamgcn": "SiamGCN"}
# Models whose training failed the AUC gate; flagged in the forest plot.
FAILED = {("urb3dcd_v2_ms", "siamese_pointnet"), ("hkcd", "siamese_kpconv")}
TIE_EPS = 0.002          # |dF1| below this counts as a tie in the head-to-head record

OUT = os.path.join("paper", "figures")

plt.rcParams.update({"font.size": 9, "axes.edgecolor": SOFT, "axes.labelcolor": INK,
                     "xtick.color": SOFT, "ytick.color": SOFT, "figure.facecolor": "white"})


def despine(ax, keep_left=True):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if not keep_left:
        ax.spines["left"].set_visible(False)


def read_f1():
    """{(dataset, model, method): test_mean_f1} from the main + Otsu tables."""
    out = {}
    for path in ["results/main_table.csv", "results/otsu_baseline_table.csv"]:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                out[(row["dataset"], row["model"], row["method"])] = float(row["test_mean_f1"])
    return out


def combos_of(f1):
    """The 22 model x dataset combinations, in reading order."""
    return [(ds, m) for ds in DS_ORDER for m in MODEL_ORDER if (ds, m, "f1_optimal") in f1]


def fig_head_to_head(f1):
    """(a) win/tie/loss of ours against each baseline; (b) mean rank of all seven."""
    combos = combos_of(f1)
    # Sized close to the 6.2 in text width so the document barely downscales it.
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7),
                             gridspec_kw={"width_ratios": [1.32, 1.0]})

    # --- (a) head-to-head record -------------------------------------------------
    ax = axes[0]
    y = np.arange(len(BASELINES))
    for i, base in enumerate(BASELINES):
        diffs = np.array([f1[c + ("quantile_shrink",)] - f1[c + (base,)] for c in combos])
        w = int(np.sum(diffs > TIE_EPS))
        t = int(np.sum(np.abs(diffs) <= TIE_EPS))
        l = int(np.sum(diffs < -TIE_EPS))
        left = 0.0
        for count, color in [(w, WIN), (t, TIE), (l, LOSS)]:
            if count:
                ax.barh(i, count, left=left, height=0.62, color=color,
                        edgecolor="white", linewidth=1.2)
                ax.text(left + count / 2.0, i, str(count), ha="center", va="center",
                        fontsize=8.5, color="white" if color != TIE else INK,
                        fontweight="bold")
            left += count
    ax.set_yticks(y)
    ax.set_yticklabels([M_LABEL[b] for b in BASELINES], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlim(0, len(combos))
    ax.set_xlabel("number of the 22 model x dataset combinations", fontsize=8.2)
    ax.set_title("(a)  quantile-shrink against each baseline", fontsize=9.5, color=INK, loc="left")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (WIN, TIE, LOSS)]
    ax.legend(handles, ["ours better", "tie (< 0.002 F1)", "ours worse"], frameon=False,
              fontsize=7.8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.30))
    despine(ax)

    # --- (b) mean rank -----------------------------------------------------------
    ax = axes[1]
    ranks = {m: [] for m in METHODS7}
    for c in combos:
        vals = np.array([f1[c + (m,)] for m in METHODS7])
        for m, r in zip(METHODS7, rankdata(-vals, method="average")):   # 1 = best F1
            ranks[m].append(r)
    order = sorted(METHODS7, key=lambda m: np.mean(ranks[m]))
    vals = [float(np.mean(ranks[m])) for m in order]
    yy = np.arange(len(order))
    ax.barh(yy, vals, height=0.62, color=[M_COLOR[m] for m in order])
    for i, v in enumerate(vals):
        ax.text(v + 0.08, i, "%.2f" % v, va="center", fontsize=8, color=INK)
    ax.set_yticks(yy)
    ax.set_yticklabels([M_LABEL[m] for m in order], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 7.4)
    ax.set_xlabel("mean rank  (1 = best)", fontsize=8.5)
    ax.set_title("(b)  average rank of every rule", fontsize=9.5, color=INK, loc="left")
    ax.grid(axis="x", color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    despine(ax)

    fig.tight_layout(w_pad=1.8)
    fig.savefig(os.path.join(OUT, "fig_head_to_head.png"), dpi=600, bbox_inches="tight")
    fig.savefig(os.path.join(OUT, "fig_head_to_head.pdf"), bbox_inches="tight")
    plt.close(fig)


def fig_forest():
    """Per-combination dF1 vs the tuned global threshold, with bootstrap 95% CIs."""
    rows = {}
    with open("results/significance_table.csv", newline="") as f:
        for r in csv.DictReader(f):
            if r["comparison"] == "quantile_shrink_vs_f1_optimal" and r["dataset"] != "ALL_COMBOS":
                rows[(r["dataset"], r["model"])] = r
    order = [(ds, m) for ds in DS_ORDER for m in MODEL_ORDER if (ds, m) in rows]

    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    labels = []
    for i, key in enumerate(order):
        r = rows[key]
        y = len(order) - 1 - i                             # top row = first combo
        d = float(r["delta"])
        lo, hi = float(r["boot_ci_lo"]), float(r["boot_ci_hi"])
        sig = float(r["boot_p"]) < 0.05
        color = (WIN if d > 0 else LOSS) if sig else "#9b9a96"
        ax.plot([lo, hi], [y, y], color=color, lw=2.0, solid_capstyle="round", zorder=2)
        ax.plot([d], [y], marker="o", ms=6.0, color=color, zorder=3,
                markeredgecolor="white", markeredgewidth=0.8)
        tag = "  (training failed)" if key in FAILED else ""
        labels.append("%s %s%s" % (DS_SHORT[key[0]], MODEL_LABEL[key[1]], tag))
    ax.axvline(0.0, color=INK, lw=1.0, zorder=1)
    # Light separators between datasets.
    counts, run = [], 0
    for ds in DS_ORDER:
        n = sum(1 for k in order if k[0] == ds)
        if n:
            run += n
            counts.append(run)
    for c in counts[:-1]:
        ax.axhline(len(order) - c - 0.5, color=GRID, lw=0.9, zorder=0)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(labels[::-1], fontsize=8.2)
    ax.set_ylim(-0.8, len(order) - 0.2)
    ax.set_xlabel("change in test mean per-scene F1  (quantile-shrink minus tuned global threshold)",
                  fontsize=8.6)
    ax.set_title("Where per-cube calibration helps, and by how much\n"
                 "colored = nominally significant at uncorrected p < 0.05 "
                 "(paired bootstrap over test cubes);\n"
                 "gray = indistinguishable from the tuned global threshold",
                 fontsize=9.2, color=INK)
    ax.grid(axis="x", color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    despine(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_forest.png"), dpi=600, bbox_inches="tight")
    fig.savefig(os.path.join(OUT, "fig_forest.pdf"), bbox_inches="tight")
    plt.close(fig)


def fig_signal_bound():
    """(a) the oracle headroom survives a split-half check; (b) but the oracle
    threshold is barely predictable from label-free per-cube statistics."""
    rows = list(csv.DictReader(open("results/estimator_autopsy.csv", newline="")))
    # MS PointNet predicts change everywhere, so it has no usable regression cubes
    # (n_reg_cubes = 0) and no R^2; it is excluded from panel (b) and noted in the caption.
    reg_rows = [r for r in rows if r["rf_cv_r2_bothsides"] and r["linreg_cv_r2_bothsides"]]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))

    # --- (a) reported vs honest headroom ----------------------------------------
    ax = axes[0]
    top = 0.0
    for ds in DS_ORDER:
        xs = [float(r["reported_headroom_B"]) for r in rows if r["dataset"] == ds]
        ys = [float(r["honest_headroom_B"]) for r in rows if r["dataset"] == ds]
        ax.scatter(xs, ys, s=30, color=DS_COLOR[ds], edgecolor="white", linewidth=0.6,
                   label=DS_SHORT[ds], zorder=3)
        top = max([top] + xs + ys)
    lim = top * 1.12
    ax.plot([0, lim], [0, lim], ls="--", color=SOFT, lw=1.1, zorder=2,
            label="headroom fully survives")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("oracle headroom, in-sample", fontsize=8.4)
    ax.set_ylabel("oracle headroom, held-out half", fontsize=8.4)
    ax.set_title("(a)  the signal is real", fontsize=9.3, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=7.6, loc="upper left")
    ax.grid(color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    despine(ax)

    # --- (b) predictability of the oracle threshold ------------------------------
    ax = axes[1]
    rng = np.random.default_rng(42)
    for i, ds in enumerate(DS_ORDER):
        vals = [max(float(r["linreg_cv_r2_bothsides"]), float(r["rf_cv_r2_bothsides"]))
                for r in reg_rows if r["dataset"] == ds]
        jitter = rng.uniform(-0.10, 0.10, size=len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals, s=34, color=DS_COLOR[ds],
                   edgecolor="white", linewidth=0.6, zorder=3)
        ax.plot([i - 0.26, i + 0.26], [np.mean(vals)] * 2, color=INK, lw=1.6, zorder=4)
    ax.axhline(1.0, color=SOFT, ls="--", lw=1.1, zorder=2)
    ax.text(3.40, 0.96, "perfect prediction", ha="right", va="top", fontsize=7.4, color=SOFT)
    ax.set_xticks(range(len(DS_ORDER)))
    ax.set_xticklabels([DS_SHORT[d] for d in DS_ORDER], fontsize=8.2)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("cross-validated R² of the oracle threshold", fontsize=8.4)
    ax.set_title("(b)  but it is not predictable without labels", fontsize=9.3,
                 color=INK, loc="left")
    ax.grid(axis="y", color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    despine(ax)

    fig.tight_layout(w_pad=2.2)
    fig.savefig(os.path.join(OUT, "fig_signal_bound.png"), dpi=600, bbox_inches="tight")
    fig.savefig(os.path.join(OUT, "fig_signal_bound.pdf"), bbox_inches="tight")
    plt.close(fig)
    best = max(max(float(r["linreg_cv_r2_bothsides"]), float(r["rf_cv_r2_bothsides"]))
               for r in reg_rows)
    print("best R^2 over %d combos: %.3f" % (len(reg_rows), best))


def fig_sensitivity():
    """(a) the validation optimum is broad and the test result moves little inside it;
    (b) alpha and k can be shared across every pair, only the offset c needs refitting."""
    rows = list(csv.DictReader(open("results/sensitivity_table.csv", newline="")))
    rows.sort(key=lambda r: (DS_ORDER.index(r["dataset"]), MODEL_ORDER.index(r["model"])))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.0))

    # --- (a) the same candidates, measured against the tuned global threshold -----
    # Plotting the difference (not the absolute F1) puts every combination on one
    # scale and makes the zero line the thing the text actually compares against.
    ax = axes[0]
    for i, r in enumerate(rows):
        g = float(r["test_global"])
        lo, hi = float(r["test_near_p05"]) - g, float(r["test_near_p95"]) - g
        color = DS_COLOR[r["dataset"]]
        ax.plot([lo, hi], [i, i], color=color, lw=3.0, alpha=0.40, solid_capstyle="round",
                zorder=2)
        ax.scatter([float(r["test_deployed"]) - g], [i], s=24, color=color, zorder=4,
                   edgecolor="white", linewidth=0.5)
    ax.axvline(0.0, color=INK, lw=1.0, zorder=3)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(["%s %s" % (DS_SHORT[r["dataset"]], MODEL_LABEL[r["model"]])
                        for r in rows], fontsize=6.4)
    ax.invert_yaxis()
    ax.set_xlim(-0.09, 0.09)
    ax.set_xlabel("test F1 minus the tuned global threshold", fontsize=8.4)
    ax.set_title("(a)  the validation optimum is broad", fontsize=9.3, color=INK, loc="left")
    handles = [plt.Line2D([], [], color=SOFT, lw=3.0, alpha=0.5,
                          label="candidates within 0.01 val F1 (5-95%)"),
               plt.Line2D([], [], marker="o", ls="", ms=4.5, color=SOFT, label="deployed fit"),
               plt.Line2D([], [], color=INK, lw=1.0, label="tuned global threshold")]
    ax.legend(handles=handles, frameon=False, fontsize=6.6, ncol=1,
              loc="upper center", bbox_to_anchor=(0.5, -0.16))
    ax.grid(axis="x", color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    despine(ax)

    # --- (b) sharing alpha and k across every model and dataset ------------------
    ax = axes[1]
    lo = min(float(r["test_shared"]) for r in rows) - 0.04
    hi = max(float(r["test_shared"]) for r in rows) + 0.04
    ax.plot([lo, hi], [lo, hi], ls="--", color=SOFT, lw=1.1, zorder=2,
            label="identical")
    for ds in DS_ORDER:
        xs = [float(r["test_deployed"]) for r in rows if r["dataset"] == ds]
        ys = [float(r["test_shared"]) for r in rows if r["dataset"] == ds]
        ax.scatter(xs, ys, s=30, color=DS_COLOR[ds], edgecolor="white", linewidth=0.6,
                   zorder=3, label=DS_SHORT[ds])
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("all three scalars fitted per pair", fontsize=8.4)
    ax.set_ylabel("$\\alpha$ and $k$ shared, only $c$ refitted", fontsize=8.4)
    ax.set_title("(b)  only the offset needs refitting", fontsize=9.3, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=7.0, loc="upper left")
    ax.grid(color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    despine(ax)

    fig.tight_layout(w_pad=2.0)
    fig.savefig(os.path.join(OUT, "fig_sensitivity.png"), dpi=600, bbox_inches="tight")
    fig.savefig(os.path.join(OUT, "fig_sensitivity.pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    f1 = read_f1()
    fig_head_to_head(f1)
    fig_forest()
    fig_signal_bound()
    fig_sensitivity()
    print("wrote paper/figures/fig_{head_to_head,forest,signal_bound,sensitivity}.png")


if __name__ == "__main__":
    main()
