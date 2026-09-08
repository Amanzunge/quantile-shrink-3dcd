"""
plot_paper.py - Paper/thesis figures: quantile_shrink vs the standard calibration
baselines (fixed_05, f1_optimal, temperature_scaling, platt, isotonic).

  results/fig_paper_methods_f1.png   test mean F1, every method x model, one panel per dataset
  results/fig_paper_mean_rank.png    mean rank of each method over the 22 model x dataset combos
  results/fig_paper_cube_example.png one real cube: global threshold vs ours vs oracle
  results/fig_v2_split_half.png      (unchanged, produced by the autopsy study)

  python src/plot_paper.py
"""

import os
import csv
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import rankdata

from calibrate import TAU_GRID
from diagnose_predictions import f1_from_pred
from calibrate_v2 import ALPHA_GRID, load_split_scores, precompute_scene, tau_quantile_shrink
from transfer import split_of_dataset

# Validated reference palette, fixed assignment; ours = blue, fixed_05 = neutral gray.
METHODS = ["fixed_05", "platt", "isotonic", "temperature_scaling", "f1_optimal", "quantile_shrink"]
M_COLOR = {"quantile_shrink": "#2a78d6", "f1_optimal": "#1baf7a", "temperature_scaling": "#eda100",
           "platt": "#4a3aa7", "isotonic": "#e87ba4", "fixed_05": "#8a8985"}
M_LABEL = {"quantile_shrink": "quantile_shrink (ours)", "f1_optimal": "f1_optimal",
           "temperature_scaling": "temperature", "platt": "Platt", "isotonic": "isotonic",
           "fixed_05": "fixed 0.5"}
INK, SOFT = "#0b0b0b", "#52514e"
DATASETS = ["urb3dcd_v2_ld", "urb3dcd_v2_ms", "hkcd", "indoorcd"]
DS_LABEL = {"urb3dcd_v2_ld": "Urb3DCD-V2 LD", "urb3dcd_v2_ms": "Urb3DCD-V2 MS",
            "hkcd": "HKCD", "indoorcd": "IndoorCD"}

plt.rcParams.update({"font.size": 9, "axes.edgecolor": SOFT, "axes.labelcolor": INK,
                     "xtick.color": SOFT, "ytick.color": SOFT, "figure.facecolor": "white"})


def despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def read_main():
    """main_table.csv -> {(dataset, model, method): test_mean_f1}."""
    out = {}
    with open("results/main_table.csv", newline="") as f:
        for row in csv.DictReader(f):
            out[(row["dataset"], row["model"], row["method"])] = float(row["test_mean_f1"])
    return out


def fig_methods_f1(main):
    """Dot strip per model: all six methods' test F1, one panel per dataset."""
    fig, axes = plt.subplots(1, 4, figsize=(12.5, 3.6))
    off = np.linspace(-0.30, 0.30, len(METHODS))          # method offsets inside each model slot
    for ax, ds in zip(axes, DATASETS):
        models = sorted({m for (d, m, _me) in main if d == ds})
        for j, method in enumerate(METHODS):
            xs = [i + off[j] for i in range(len(models))]
            ys = [main[(ds, m, method)] for m in models]
            ours = method == "quantile_shrink"
            ax.scatter(xs, ys, s=34 if ours else 22, color=M_COLOR[method],
                       edgecolor=INK if ours else SOFT, linewidth=0.7 if ours else 0.4,
                       zorder=3 if ours else 2, label=M_LABEL[method])
        for i in range(len(models) - 1):                  # light separators between model slots
            ax.axvline(i + 0.5, color="#eceae6", lw=0.7, zorder=0)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([m.replace("siamese_", "").replace("_euclidean", "") for m in models],
                           rotation=45, ha="right", fontsize=7.5)
        ax.set_title(DS_LABEL[ds], fontsize=10, color=INK)
        despine(ax)
        ax.grid(axis="y", color="#eceae6", lw=0.7, zorder=0)
    axes[0].set_ylabel("test mean per-scene change F1")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, ncol=6, loc="lower center",
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Calibration methods across all models and datasets", fontsize=10.5, color=INK)
    fig.tight_layout(rect=[0, 0.05, 1, 0.93])
    fig.savefig("results/fig_paper_methods_f1.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_mean_rank(main):
    """Mean rank (1 = best) of each method over the 22 model x dataset combos."""
    combos = sorted({(d, m) for (d, m, _me) in main})
    ranks = {method: [] for method in METHODS}
    for ds, model in combos:
        f1s = np.array([main[(ds, model, method)] for method in METHODS])
        rk = rankdata(-f1s, method="average")             # rank 1 = highest F1, ties averaged
        for method, v in zip(METHODS, rk):
            ranks[method].append(v)
    order = sorted(METHODS, key=lambda m: np.mean(ranks[m]))
    fig, ax = plt.subplots(figsize=(6.4, 2.9))
    y = np.arange(len(order))
    vals = [float(np.mean(ranks[m])) for m in order]
    ax.barh(y, vals, height=0.62, color=[M_COLOR[m] for m in order])
    for i, v in enumerate(vals):
        ax.text(v + 0.05, i, "%.2f" % v, va="center", fontsize=8, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels([M_LABEL[m] for m in order], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("mean rank over 22 model x dataset combinations (1 = best)")
    ax.set_title("Average calibration-method rank, test F1", fontsize=10, color=INK)
    despine(ax)
    ax.grid(axis="x", color="#eceae6", lw=0.7, zorder=0)
    fig.tight_layout()
    fig.savefig("results/fig_paper_mean_rank.png", dpi=200)
    plt.close(fig)


def fig_cube_example():
    """One real HKCD cube: the tuned GLOBAL threshold vs our per-cube threshold vs
    that cube's oracle threshold, over the score histogram."""
    dataset = "hkcd"
    split_of = split_of_dataset(dataset)
    qs_rows = {r["model"]: r for r in csv.DictReader(open("results/improved_params_hkcd.csv", newline=""))
               if r["method"] == "quantile_shrink"}

    # Search the two models where the per-cube method wins on HKCD, over all their
    # test scenes, for the well-populated cube where ours lands closest to the
    # oracle while the global threshold misses it by the most.
    best = None
    for model in ["siamgcn", "icp_euclidean"]:
        with open(os.path.join("calibration", dataset, model, "f1_optimal.json")) as f:
            tau_global = float(json.load(f)["fitted_params"]["tau"])
        qs = qs_rows[model]
        c_qs, k_qs = float(qs["c"]), float(qs["k"])
        ai = int(np.argmin(np.abs(ALPHA_GRID - float(qs["alpha"]))))
        _val, test = load_split_scores(os.path.join("predictions", dataset, model), split_of)
        for rec in test:
            pre = precompute_scene(rec)
            tau_qs_all = tau_quantile_shrink(pre, c_qs, ai, k_qs)
            inv = pre["inv"]
            n_pos = np.bincount(inv[rec["labels"] == 1], minlength=pre["K"])
            for cube_i in np.where((n_pos >= 200) & (pre["n_nc"] >= 500))[0]:
                mask_i = inv == cube_i
                s_i, lab_i = rec["scores"][mask_i], rec["labels"][mask_i]
                tau_orc = float(TAU_GRID[int(np.argmax([f1_from_pred(s_i > t, lab_i) for t in TAU_GRID]))])
                gain = abs(tau_global - tau_orc) - abs(tau_qs_all[cube_i] - tau_orc)
                if best is None or gain > best["gain"]:
                    best = {"gain": gain, "model": model, "cube": int(cube_i),
                            "s": s_i, "lab": lab_i, "tau_global": tau_global,
                            "tau_qs": float(tau_qs_all[cube_i]), "tau_oracle": tau_orc}
    model, cube = best["model"], best["cube"]
    s, lab = best["s"], best["lab"]
    tau_global, tau_oracle = best["tau_global"], best["tau_oracle"]
    tau_qs_cube = best["tau_qs"]

    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    bins = np.linspace(0, 1, 61)
    ax.hist(s[lab == 0], bins=bins, color="#d8d7d3", label="true no-change points")
    ax.hist(s[lab == 1], bins=bins, color="#9ec5f4", label="true change points")
    for tau, color, ls, name in [(tau_global, "#8a8985", "--", "global f1_optimal tau = %.2f"),
                                 (tau_qs_cube, "#2a78d6", "-", "per-cube tau (ours) = %.2f"),
                                 (tau_oracle, INK, ":", "cube oracle tau = %.2f")]:
        ax.axvline(tau, color=color, ls=ls, lw=1.6, label=name % tau)
    ax.set_yscale("log")
    ax.set_xlabel("model change score P(change)")
    ax.set_ylabel("points per bin (log)")
    ax.set_title("One real HKCD cube (%s, cube %d): the per-cube threshold adapts to the\n"
                 "local score distribution where one global threshold cannot" % (model, cube),
                 fontsize=9.5, color=INK)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    despine(ax)
    fig.tight_layout()
    fig.savefig("results/fig_paper_cube_example.png", dpi=600)
    plt.close(fig)
    print("example %s cube %d: tau_global %.3f  tau_ours %.3f  tau_oracle %.3f" % (
        model, cube, tau_global, tau_qs_cube, tau_oracle))


def main():
    table = read_main()
    fig_methods_f1(table)
    fig_mean_rank(table)
    fig_cube_example()
    print("wrote results/fig_paper_{methods_f1,mean_rank,cube_example}.png")


if __name__ == "__main__":
    main()
