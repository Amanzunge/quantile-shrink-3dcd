"""
plot_presubmission.py - The three NEW paper figures (draft v1), written to paper/figures/:

  fig_cd_diagram.png        Demsar critical-difference diagram: mean rank of the six
                            paper methods + the otsu_cube baseline over the 22
                            model x dataset combos, Nemenyi CD at alpha = 0.05.
  fig_method_schematic.png  Pipeline + per-cube threshold formula (paper Fig. 1).
  fig_qualitative_hkcd.png  One HKCD test scene, error map under the tuned global
                            threshold vs the per-cube quantile_shrink threshold.

Also copies the three REUSED figures from results/ into paper/figures/.

Torch-free, local CPU:  python src/plot_presubmission.py
"""

import os
import csv
import json
import shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from scipy.stats import rankdata

# Same palette and ink colors as plot_paper.py so all figures match.
M_COLOR = {"quantile_shrink": "#2a78d6", "f1_optimal": "#1baf7a", "temperature_scaling": "#eda100",
           "platt": "#4a3aa7", "isotonic": "#e87ba4", "fixed_05": "#8a8985", "otsu_cube": "#b0552a"}
M_LABEL = {"quantile_shrink": "quantile_shrink (ours)", "f1_optimal": "f1_optimal",
           "temperature_scaling": "temperature", "platt": "Platt", "isotonic": "isotonic",
           "fixed_05": "fixed 0.5", "otsu_cube": "per-cube Otsu"}
INK, SOFT, GRID = "#0b0b0b", "#52514e", "#eceae6"
METHODS7 = ["fixed_05", "platt", "isotonic", "temperature_scaling", "f1_optimal",
            "otsu_cube", "quantile_shrink"]

plt.rcParams.update({"font.size": 9, "axes.edgecolor": SOFT, "axes.labelcolor": INK,
                     "xtick.color": SOFT, "ytick.color": SOFT, "figure.facecolor": "white"})

OUT = os.path.join("paper", "figures")


def read_f1():
    """{(dataset, model, method): test_mean_f1} from the main + otsu tables."""
    out = {}
    for path in ["results/main_table.csv", "results/otsu_baseline_table.csv"]:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                out[(row["dataset"], row["model"], row["method"])] = float(row["test_mean_f1"])
    return out


def fig_cd_diagram(f1):
    """Classic Demsar CD diagram over the 22 combos, 7 methods, Nemenyi alpha=0.05."""
    combos = sorted({(d, m) for (d, m, _me) in f1})
    ranks = []
    for ds, model in combos:
        vals = np.array([f1[(ds, model, meth)] for meth in METHODS7])
        ranks.append(rankdata(-vals, method="average"))   # rank 1 = best F1, ties averaged
    mean_rank = np.mean(ranks, axis=0)
    k, n = len(METHODS7), len(combos)
    q_alpha = 2.949                                       # studentized range q_0.05 for k=7 (Demsar Tab.5)
    cd = q_alpha * np.sqrt(k * (k + 1) / (6.0 * n))

    order = np.argsort(mean_rank)                         # best first
    names = [M_LABEL[METHODS7[i]] for i in order]
    mr = mean_rank[order]

    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    lo, hi = 1.0, float(k)
    ax.set_xlim(lo - 0.15, hi + 0.15)
    ax.set_ylim(-3.6, 2.2)
    ax.plot([lo, hi], [0, 0], color=INK, lw=1.2)          # the rank axis
    for t in range(1, k + 1):
        ax.plot([t, t], [0, 0.12], color=INK, lw=1.0)
        ax.text(t, 0.30, str(t), ha="center", fontsize=8.5, color=INK)
    # CD ruler above the axis.
    ax.plot([lo, lo + cd], [1.5, 1.5], color=INK, lw=1.6)
    ax.plot([lo, lo], [1.40, 1.60], color=INK, lw=1.6)
    ax.plot([lo + cd, lo + cd], [1.40, 1.60], color=INK, lw=1.6)
    ax.text(lo + cd / 2, 1.72, "CD = %.2f  (Nemenyi, alpha = 0.05)" % cd,
            ha="center", fontsize=8.5, color=INK)
    # Method stems: best half named on the left margin, worst half on the right.
    half = int(np.ceil(k / 2))
    for pos, (name, r) in enumerate(zip(names, mr)):
        left = pos < half
        row = pos if left else (k - 1 - pos)              # vertical slot on its own side
        y_name = -1.1 - 0.62 * row
        x_name = lo - 0.12 if left else hi + 0.12
        meth = METHODS7[order[pos]]
        lw = 2.2 if meth == "quantile_shrink" else 1.2
        color = M_COLOR[meth] if meth == "quantile_shrink" else SOFT
        ax.plot([r, r], [0, y_name], color=color, lw=lw, zorder=2)          # vertical drop
        ax.plot([r, x_name], [y_name, y_name], color=color, lw=lw, zorder=2)  # horizontal lead
        weight = "bold" if meth == "quantile_shrink" else "normal"
        ax.text(x_name - 0.03 if left else x_name + 0.03, y_name,
                "%s (%.2f)" % (name, r), ha="right" if left else "left", va="center",
                fontsize=8.5, color=INK, fontweight=weight)
    # Group bars: maximal cliques of methods whose rank spread is within the CD.
    cliques = []
    for i in range(k):
        j = i
        while j + 1 < k and mr[j + 1] - mr[i] <= cd:
            j += 1
        if j > i and not any(a <= i and j <= b for a, b in cliques):
            cliques.append((i, j))
    for level, (a, b) in enumerate(cliques):
        y = -0.28 - 0.30 * level
        ax.plot([mr[a] - 0.04, mr[b] + 0.04], [y, y], color=INK, lw=3.0,
                solid_capstyle="round", alpha=0.85)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_cd_diagram.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("CD diagram: N=%d, CD=%.3f" % (n, cd))
    for name, r in zip(names, mr):
        print("  %-24s %.2f" % (name, r))


def box(ax, x, y, w, h, text, fc="#f4f3f0", fontsize=8.2):
    """One rounded pipeline box with centred text."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012",
                                fc=fc, ec=SOFT, lw=1.0))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color=INK)


def arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=11, color=SOFT, lw=1.2))


def fig_method_schematic():
    """Paper Fig. 1: the cube pipeline (top row) + the calibrated per-cube
    threshold (bottom band) with the fitting protocol."""
    fig, ax = plt.subplots(figsize=(8.6, 3.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    # Top row: the frozen prediction pipeline (identical for every model).
    steps = ["bi-temporal\npoint clouds\n$t_0$, $t_1$",
             "tile into\ncubes\n(50 m grid)",
             "FPS to model\nnative size\n(train: random,\ntest: fixed)",
             "Siamese\nnetwork\n(per-point\nscores)",
             "NN-propagate\nto full\nresolution"]
    xw, gap = 0.155, 0.036
    x0, yt, ht = 0.015, 0.62, 0.33
    for i, s in enumerate(steps):
        box(ax, x0 + i * (xw + gap), yt, xw, ht, s)
        if i:
            arrow(ax, x0 + i * (xw + gap) - gap + 0.006, yt + ht / 2,
                  x0 + i * (xw + gap) - 0.004, yt + ht / 2)
    # Decision box, visually ours.
    ax.add_patch(FancyBboxPatch((0.015, 0.10), 0.60, 0.34, boxstyle="round,pad=0.012",
                                fc="#e8f0fb", ec=M_COLOR["quantile_shrink"], lw=1.6))
    ax.text(0.315, 0.375, "per-cube decision:  point is change  $\\Leftrightarrow$  score $> \\tau_i$",
            ha="center", fontsize=9.0, color=INK)
    ax.text(0.315, 0.265,
            r"$\tau_i = \mathrm{clip}_{[0,1]}\!\left(c + w_i\, \hat{q}_\alpha(\mathcal{N}_i) + (1-w_i)\, \hat{q}_\alpha(\mathcal{N}_{scene})\right)$,"
            r"   $w_i = \frac{n_i}{n_i + k}$",
            ha="center", fontsize=9.6, color=INK)
    ax.text(0.315, 0.18,
            r"$\mathcal{N}_i$: cube $i$ scores predicted no-change (score $\leq$ 0.5, no labels)",
            ha="center", fontsize=7.6, color=SOFT)
    ax.text(0.315, 0.125, r"$q_\alpha$: empirical quantile at conformal rank $\lceil \alpha (n{+}1) \rceil$",
            ha="center", fontsize=7.6, color=SOFT)
    # Fit box on the right.
    box(ax, 0.655, 0.10, 0.33, 0.34,
        "$(c,\\ \\alpha,\\ k)$: three scalars,\nfit ONCE on validation scenes\n"
        "by grid search maximising\nmean per-scene change F1;\nfrozen at test time", fc="#f4f3f0")
    arrow(ax, 0.86, 0.62, 0.50, 0.46)                     # full-resolution scores -> decision
    arrow(ax, 0.655, 0.27, 0.627, 0.27)                   # fit -> decision
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_method_schematic.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def fig_qualitative():
    """HKCD siamgcn, the test scene where quantile_shrink helps most: top-down error
    maps under the tuned global tau (left) and the per-cube tau (right)."""
    dataset, model = "hkcd", "siamgcn"
    with open(os.path.join("calibration", dataset, model, "f1_optimal.json")) as f:
        fo = json.load(f)
    with open(os.path.join("calibration_v2", dataset, model, "quantile_shrink.json")) as f:
        qs = json.load(f)
    # Pick the test scene with the largest per-scene F1 gain.
    scenes = [s for s, v in fo["per_scene"].items() if v["split"] == "test"]
    scene = max(scenes, key=lambda s: qs["per_scene"][s]["f1"] - fo["per_scene"][s]["f1"])
    tau_global = float(fo["fitted_params"]["tau"])
    tau_cube = np.array(qs["per_scene"][scene]["tau"], dtype=np.float64)

    d = np.load(os.path.join("predictions", dataset, model, scene + ".npz"), allow_pickle=True)
    scores = d["scores"].astype(np.float64)
    labels = d["labels"].astype(np.int64)
    coords = d["coords"].astype(np.float64)
    cubes, inv = np.unique(d["cube_id"], return_inverse=True)

    pred_g = scores > tau_global
    pred_q = scores > tau_cube[inv]

    rng = np.random.default_rng(42)
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 5.0), sharex=True, sharey=True)
    xy = coords[:, :2] - coords[:, :2].mean(axis=0)       # centre for readable axes
    for ax, pred, name, f1v in [(axes[0], pred_g, "tuned global threshold (f1_optimal)",
                                 fo["per_scene"][scene]["f1"]),
                                (axes[1], pred_q, "per-cube quantile_shrink (ours)",
                                 qs["per_scene"][scene]["f1"])]:
        tn = ~pred & (labels == 0)
        tp = pred & (labels == 1)
        fp = pred & (labels == 0)
        fn = ~pred & (labels == 1)
        # Background: subsample the (dominant) true-negative points hard.
        for mask, color, size, cap, label in [
                (tn, "#e3e2de", 0.25, 250000, None),
                (tp, "#9ec5f4", 0.35, 250000, "true positive"),
                (fp, "#eda100", 0.45, 250000, "false positive"),
                (fn, "#c9366b", 0.45, 250000, "false negative")]:
            idx = np.where(mask)[0]
            if len(idx) > cap:
                idx = rng.choice(idx, cap, replace=False)
            ax.scatter(xy[idx, 0], xy[idx, 1], s=size, c=color, marker=".",
                       linewidths=0, rasterized=True, label=label)
        ax.set_title("%s\nscene %s, change F1 = %.3f" % (name, scene, f1v),
                     fontsize=9.5, color=INK)
        ax.set_aspect("equal")
        ax.set_xlabel("x (m)")
        ax.tick_params(labelsize=7.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("y (m)")
    handles = [plt.Line2D([], [], marker="o", ls="", ms=6, color=c, label=l)
               for c, l in [("#9ec5f4", "true positive"), ("#eda100", "false positive"),
                            ("#c9366b", "false negative"), ("#e3e2de", "true negative")]]
    fig.legend(handles=handles, frameon=False, ncol=4, loc="lower center", fontsize=8.5,
               bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(os.path.join(OUT, "fig_qualitative_hkcd.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("qualitative: scene %s  global F1 %.4f -> ours %.4f" % (
        scene, fo["per_scene"][scene]["f1"], qs["per_scene"][scene]["f1"]))


def fig_equations():
    """The three numbered display equations of Section 3.2, rendered at 300 dpi so
    Word shows real typeset math. Fixed 6.5 in width, equation centred, number right."""
    eqs = [
        ("eq1.png", 0.45,
         r"$\mathcal{N}_i \;=\; \{\, p \in C_i \;:\; s_p \leq \frac{1}{2} \,\}, \qquad n_i = |\mathcal{N}_i|$", "(1)"),
        ("eq2.png", 0.40,
         r"$\hat{q}_\alpha(\mathcal{N}_i) \;=\; s_{(r)}, \quad r = \left\lceil \alpha\,(n_i + 1) \right\rceil,"
         r" \qquad \mathrm{P}\!\left( s > \hat{q}_\alpha(\mathcal{N}_i) \right) \;\leq\; 1 - \alpha$", "(2)"),
        ("eq3.png", 0.55,
         r"$\tau_i \;=\; \mathrm{clip}_{[0,1]}\!\left( c \,+\, w_i\, \hat{q}_\alpha(\mathcal{N}_i)"
         r" \,+\, (1 - w_i)\, \hat{q}_\alpha(\mathcal{N}_{\mathrm{scene}}) \right),"
         r" \qquad w_i = \frac{n_i}{n_i + k}$", "(3)"),
    ]
    for name, h, tex, num in eqs:
        fig = plt.figure(figsize=(6.5, h))
        fig.patch.set_facecolor("white")
        fig.text(0.5, 0.5, tex, ha="center", va="center", fontsize=13, color=INK)
        fig.text(0.985, 0.5, num, ha="right", va="center", fontsize=11, color=INK)
        fig.savefig(os.path.join(OUT, name), dpi=300)
        plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    fig_equations()
    f1 = read_f1()
    fig_cd_diagram(f1)
    fig_method_schematic()
    fig_qualitative()
    # Reused figures, copied so paper/ is self-contained.
    for name in ["fig_paper_cube_example.png", "fig_gain_vs_headroom_scatter.png",
                 "fig_cube_geometry_boundary.png"]:
        shutil.copy(os.path.join("results", name), os.path.join(OUT, name))
    print("wrote 3 new + copied 3 reused figures into paper/figures/")


if __name__ == "__main__":
    main()
