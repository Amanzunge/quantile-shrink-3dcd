"""
Task 12 synthesis figures, in the paper visual style of src/plot_paper.py (same
palette, despined axes, light grid, dpi 200). Since the 2026-07-14 paper-layer
decision the per-cube method shown is the HEADLINE quantile_shrink, read from the
quantile_shrink_* columns of the ablation CSVs (two_moment is gone from the paper).

  fig_cube_geometry_boundary.png   model-free: zero-change cube % and median cubes/scene
                                   vs cube extent, all four datasets on one log-extent axis.
                                   This is the CONTINUOUS scope-limit boundary -- shrinking
                                   outdoor cubes walk toward the IndoorCD room-scale floor.
  fig_cube_gain_headroom.png       per dataset: per-cube ORACLE headroom (signal that exists)
                                   vs the REALISED quantile_shrink gain, across cube sizes. The
                                   thesis figure: headroom grows as cubes shrink while the
                                   realised gain stays near zero -> even the best estimator
                                   captures little of the per-cube signal at small scale.
  fig_fps_sweep.png                LD/MS: quantile_shrink gain and oracle headroom vs FPS size.
  fig_gain_vs_headroom_scatter.png every run as one point: headroom vs realised gain.
  fig_cube_auc_by_model.png        per-model pooled test AUC vs cube extent (the H1 gate).
  fig_cube_absolute_f1.png         absolute F1: oracle ceiling vs global tau vs ours.

Robust to PARTIAL data: it plots whatever rows exist. Torch-free, Agg backend.

Run:  python src/plot_ablations.py
"""

import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")                                    # headless: write PNGs, no display
import matplotlib.pyplot as plt

# Same reference palette and axis conventions as src/plot_paper.py.
INK, SOFT, GRID = "#0b0b0b", "#52514e", "#eceae6"
OURS = "#2a78d6"                                         # quantile_shrink, blue everywhere
GLOBAL = "#1baf7a"                                       # tuned global f1_optimal
ORACLE = "#4a3aa7"                                       # per-cube oracle ceiling/headroom

plt.rcParams.update({"font.size": 9, "axes.edgecolor": SOFT, "axes.labelcolor": INK,
                     "xtick.color": SOFT, "ytick.color": SOFT, "figure.facecolor": "white"})

DS_LABEL = {"urb3dcd_v2_ld": "Urb3DCD-V2 LD", "urb3dcd_v2_ms": "Urb3DCD-V2 MS",
            "hkcd": "HKCD", "indoorcd": "IndoorCD"}
DS_COLOR = {"urb3dcd_v2_ld": "#2a78d6", "urb3dcd_v2_ms": "#1baf7a",
            "hkcd": "#eda100", "indoorcd": "#e87ba4"}
DS_ORDER = ["urb3dcd_v2_ld", "urb3dcd_v2_ms", "hkcd", "indoorcd"]

# Per-model styling for the by-model figure, drawn from the same paper palette.
MODEL_ORDER = ["siamese_pointnet", "siamese_pointnet2", "siamese_kpconv", "siamgcn", "randla",
               "icp_euclidean"]
MODEL_LABEL = {"siamese_pointnet": "PointNet", "siamese_pointnet2": "PointNet++",
               "siamese_kpconv": "KPConv", "siamgcn": "SiamGCN", "randla": "RandLA",
               "icp_euclidean": "ICP"}
MODEL_COLOR = {"siamese_pointnet": "#2a78d6", "siamese_pointnet2": "#1baf7a",
               "siamese_kpconv": "#eda100", "siamgcn": "#4a3aa7", "randla": "#e87ba4",
               "icp_euclidean": "#8a8985"}


def despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def style(ax, which="major"):
    """Shared paper-style axis treatment: no top/right spines, light grid behind data."""
    despine(ax)
    ax.grid(True, color=GRID, lw=0.7, zorder=0, which=which)


def read_csv(path):
    """Return list-of-dicts, or [] if the file is not there yet."""
    if not os.path.exists(path):
        return []
    return list(csv.DictReader(open(path)))


def fnum(row, key):
    """Float of a CSV cell, or None for blank/missing (partial-data safe)."""
    v = row.get(key, "")
    if v in ("", None, "nan"):
        return None
    return float(v)


def plot_geometry_boundary():
    rows = read_csv("results/ablation_cube_geometry.csv")
    if not rows:
        print("skip geometry boundary: no ablation_cube_geometry.csv")
        return
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.0, 7.2), sharex=True)
    for ds in DS_ORDER:
        pts = sorted((r for r in rows if r["dataset"] == ds),
                     key=lambda r: float(r["cube_extent_m"]))
        if not pts:
            continue
        x = [float(r["cube_extent_m"]) for r in pts]
        zero = [fnum(r, "pct_zero_change_cubes") for r in pts]
        cubes = [fnum(r, "median_cubes_per_scene") for r in pts]
        ax1.plot(x, zero, "o-", color=DS_COLOR[ds], lw=1.4, ms=4.5, label=DS_LABEL[ds], zorder=3)
        ax2.plot(x, cubes, "o-", color=DS_COLOR[ds], lw=1.4, ms=4.5, label=DS_LABEL[ds], zorder=3)
    ax1.set_ylabel("zero-change cubes (%)")
    ax1.set_title("Per-cube geometry vs cube extent (model-free scope-limit boundary)",
                  fontsize=10, color=INK)
    style(ax1)
    ax1.legend(frameon=False, fontsize=8)
    ax2.set_ylabel("median cubes / scene")
    ax2.set_yscale("log")
    ax2.set_xscale("log")
    ax2.set_xlabel("cube extent (m, log scale)  --  IndoorCD at 1-2 m is the room-scale floor")
    style(ax2, which="both")
    fig.tight_layout()
    fig.savefig("results/fig_cube_geometry_boundary.png", dpi=600)
    fig.savefig("results/fig_cube_geometry_boundary.pdf")
    plt.close(fig)
    print("wrote results/fig_cube_geometry_boundary.png")


def _mean_by_size(rows, dataset, col):
    """(sizes, mean values over available models) for one dataset+column, sorted by extent."""
    by_size = {}
    for r in rows:
        if r["dataset"] != dataset:
            continue
        v = fnum(r, col)
        if v is None:
            continue
        by_size.setdefault(float(r["value"]), []).append(v)
    sizes = sorted(by_size)
    return sizes, [float(np.mean(by_size[s])) for s in sizes]


def plot_cube_gain_headroom():
    rows = read_csv("results/ablation_cube.csv")
    if not rows:
        print("skip cube gain/headroom: no ablation_cube.csv")
        return
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2))
    for ax, ds in zip(axes.ravel(), DS_ORDER):
        sx, head = _mean_by_size(rows, ds, "oracle_headroom")
        gx, gain = _mean_by_size(rows, ds, "quantile_shrink_gain")
        if sx:
            ax.plot(sx, head, "s-", color=ORACLE, lw=1.4, ms=4.5,
                    label="oracle headroom (exists)", zorder=3)
        if gx:
            ax.plot(gx, gain, "o--", color=OURS, lw=1.4, ms=4.5,
                    label="quantile_shrink gain (realised)", zorder=3)
        ax.axhline(0.0, color=SOFT, lw=0.8)
        ax.set_title(DS_LABEL[ds], fontsize=10, color=INK)
        ax.set_xlabel("cube extent (m)")
        ax.set_ylabel("F1 over tuned global tau")
        style(ax)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Per-cube signal vs realised quantile_shrink gain across cube sizes "
                 "(mean over models)", fontsize=10.5, color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig("results/fig_cube_gain_headroom.png", dpi=200)
    plt.close(fig)
    print("wrote results/fig_cube_gain_headroom.png")


def plot_fps_sweep():
    rows = read_csv("results/ablation_fps.csv")
    if not rows:
        print("skip fps sweep: no ablation_fps.csv")
        return
    datasets = [d for d in ("urb3dcd_v2_ld", "urb3dcd_v2_ms")
                if any(r["dataset"] == d for r in rows)]
    if not datasets:
        print("skip fps sweep: no LD/MS rows")
        return
    fig, axes = plt.subplots(1, len(datasets), figsize=(5.4 * len(datasets), 4.0), squeeze=False)
    for ax, ds in zip(axes[0], datasets):
        gx, gain = _mean_by_size(rows, ds, "quantile_shrink_gain")
        hx, head = _mean_by_size(rows, ds, "oracle_headroom")
        if hx:
            ax.plot(hx, head, "s-", color=ORACLE, lw=1.4, ms=4.5,
                    label="oracle headroom (exists)", zorder=3)
        if gx:
            ax.plot(gx, gain, "o--", color=OURS, lw=1.4, ms=4.5,
                    label="quantile_shrink gain (realised)", zorder=3)
        ax.axhline(0.0, color=SOFT, lw=0.8)
        ax.set_xscale("log")
        ax.set_title(DS_LABEL[ds] + "  --  FPS sweep", fontsize=10, color=INK)
        ax.set_xlabel("FPS native size (log)")
        ax.set_ylabel("F1 over tuned global tau (mean over models)")
        style(ax, which="both")
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig("results/fig_fps_sweep.png", dpi=200)
    plt.close(fig)
    print("wrote results/fig_fps_sweep.png")


def plot_gain_vs_headroom_scatter():
    """The honest 'money' figure: every run (each model x each knob value, cube AND fps) as one
    point -- x = per-cube ORACLE headroom (signal that EXISTS), y = REALISED quantile_shrink
    gain. The y=x line is full realisation; points hugging y=0 below it mean the estimator
    captures little of the available per-cube signal, regardless of how much is there."""
    rows = read_csv("results/ablation_cube.csv") + read_csv("results/ablation_fps.csv")
    if not rows:
        print("skip gain/headroom scatter: no ablation CSVs")
        return
    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    xs_all, ys_all = [], []
    for ds in DS_ORDER:
        pts = [(fnum(r, "oracle_headroom"), fnum(r, "quantile_shrink_gain"))
               for r in rows if r["dataset"] == ds]
        pts = [(x, y) for x, y in pts if x is not None and y is not None]
        if not pts:
            continue
        xs_all += [p[0] for p in pts]
        ys_all += [p[1] for p in pts]
        ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=30, color=DS_COLOR[ds],
                   alpha=0.8, edgecolor="none", label=DS_LABEL[ds], zorder=3)
    hi = max(xs_all) * 1.05 if xs_all else 0.35
    ax.plot([0, hi], [0, hi], "--", color=SOFT, lw=1, label="full realisation (gain = headroom)")
    ax.axhline(0.0, color=INK, lw=0.8)
    xa, ya = np.array(xs_all), np.array(ys_all)
    frac_pos = float((ya > 0).mean()) if xa.size else 0.0                  # runs that beat global tau
    sel = xa > 0.02                                                        # ignore ~zero-headroom runs
    med_ratio = float(np.median(ya[sel] / xa[sel])) if sel.any() else float("nan")
    ax.set_xlabel("per-cube ORACLE headroom  (F1 recoverable over tuned global tau)")
    ax.set_ylabel("REALISED quantile_shrink gain  (F1 over tuned global tau)")
    ax.set_title("Signal available vs signal captured  (each point = one model x knob value)\n"
                 "%.0f%% of runs beat global tau; median captured = %.0f%% of the headroom"
                 % (100 * frac_pos, 100 * med_ratio), fontsize=10, color=INK)
    style(ax)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig("results/fig_gain_vs_headroom_scatter.png", dpi=600)
    fig.savefig("results/fig_gain_vs_headroom_scatter.pdf")
    plt.close(fig)
    print("wrote results/fig_gain_vs_headroom_scatter.png")


def plot_cube_auc_by_model():
    """Per-model discrimination (AUC) vs cube extent -- makes the H1 boundary explicit: some models
    fall toward chance (AUC 0.5) as cubes shrink, which the model-averaged figures hide."""
    rows = read_csv("results/ablation_cube.csv")
    if not rows:
        print("skip cube AUC by model: no ablation_cube.csv")
        return
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2))
    for ax, ds in zip(axes.ravel(), DS_ORDER):
        drows = [r for r in rows if r["dataset"] == ds]
        for m in MODEL_ORDER:
            pts = sorted(((float(r["value"]), fnum(r, "pooled_test_auc"))
                          for r in drows if r["model"] == m), key=lambda t: t[0])
            pts = [(x, y) for x, y in pts if y is not None]
            if not pts:
                continue
            ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", lw=1.4, ms=4,
                    color=MODEL_COLOR[m], label=MODEL_LABEL[m], zorder=3)
        ax.axhline(0.5, color=SOFT, lw=0.8, ls=":")                        # chance
        ax.set_title(DS_LABEL[ds], fontsize=10, color=INK)
        ax.set_xlabel("cube extent (m)")
        ax.set_ylabel("pooled test AUC")
        ax.set_ylim(0.4, 1.0)
        style(ax)
        ax.legend(frameon=False, fontsize=7)
    fig.suptitle("Model discrimination vs cube extent (some models fail the AUC gate as cubes shrink)",
                 fontsize=10.5, color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig("results/fig_cube_auc_by_model.png", dpi=200)
    plt.close(fig)
    print("wrote results/fig_cube_auc_by_model.png")


def plot_cube_absolute_f1():
    """Absolute test F1 vs cube extent: per-cube oracle CEILING, tuned global f1_optimal, and
    quantile_shrink (ours) as three lines. Shows ours sits close to the global line -- both far
    below the ceiling -- and how low the absolute F1 gets at the IndoorCD room scale."""
    rows = read_csv("results/ablation_cube.csv")
    if not rows:
        print("skip cube absolute F1: no ablation_cube.csv")
        return
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2))
    for ax, ds in zip(axes.ravel(), DS_ORDER):
        sx, ceil = _mean_by_size(rows, ds, "oracle_ceiling")
        fx, f1o = _mean_by_size(rows, ds, "f1_optimal_test")
        tx, qs = _mean_by_size(rows, ds, "quantile_shrink_test")
        if sx:
            ax.plot(sx, ceil, "^-", color=ORACLE, lw=1.4, ms=5,
                    label="per-cube oracle ceiling", zorder=3)
        if fx:
            ax.plot(fx, f1o, "s--", color=GLOBAL, lw=1.4, ms=4.5,
                    label="global f1_optimal", zorder=3)
        if tx:
            ax.plot(tx, qs, "o-", color=OURS, lw=1.4, ms=4.5,
                    label="quantile_shrink (ours)", zorder=3)
        ax.set_title(DS_LABEL[ds], fontsize=10, color=INK)
        ax.set_xlabel("cube extent (m)")
        ax.set_ylabel("mean per-scene test F1")
        style(ax)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Absolute F1: quantile_shrink and a tuned global tau vs the per-cube oracle ceiling",
                 fontsize=10.5, color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig("results/fig_cube_absolute_f1.png", dpi=200)
    plt.close(fig)
    print("wrote results/fig_cube_absolute_f1.png")


def main():
    plot_geometry_boundary()
    plot_cube_gain_headroom()
    plot_fps_sweep()
    plot_gain_vs_headroom_scatter()
    plot_cube_auc_by_model()
    plot_cube_absolute_f1()


if __name__ == "__main__":
    main()
