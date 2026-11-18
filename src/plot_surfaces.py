"""
plot_surfaces.py - Visualise the quantile_shrink validation objective.

The fit maximises mean per-scene val F1 over a 3D grid (c, alpha, k). A 3D
surface cannot be drawn directly, so each model gets three orthogonal 2D slices
through the val optimum (c*, alpha*, k*):
  (c x alpha) at k*,   (c x k) at alpha*,   (alpha x k) at c*.
Colour is val F1 minus the optimum (0 = best, clipped at -CLIP), on ONE shared
scale so every panel of every dataset is comparable. The white contour marks the
-0.01 band that src/sensitivity.py calls "indistinguishable from the winner",
the star marks the deployed argmax.

Reads results/surfaces/*.npz (python src/compute_surfaces.py first).
Writes results/fig_surface_<dataset>.png, one figure per dataset.
"""

import os
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")                                    # headless: write PNGs, no display
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from plot_ablations import INK, SOFT, DS_LABEL, DS_ORDER, MODEL_ORDER, MODEL_LABEL

CLIP = 0.10          # colour floor: anything worse than optimum - 0.10 is drawn flat
BAND = 0.01          # same near-optimal band as src/sensitivity.py

# Single-hue sequential blue ramp (paper palette), light = far below optimum, dark = optimum.
CMAP = LinearSegmentedColormap.from_list(
    "blue_seq", ["#f4f8fe", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])


def draw_slice(ax, grid, x_vals, y_vals, x_label, y_label, x_ticks, y_ticks, star):
    """One heatmap panel. grid is (n_y, n_x) F1 deficit; axes are drawn on index
    positions because the alpha and k grids are not evenly spaced."""
    image = ax.imshow(grid, origin="lower", aspect="auto", cmap=CMAP,
                      vmin=-CLIP, vmax=0.0, interpolation="nearest",
                      extent=[-0.5, len(x_vals) - 0.5, -0.5, len(y_vals) - 0.5])
    # contour of the near-optimal band (needs at least 2 rows and columns)
    ax.contour(np.arange(len(x_vals)), np.arange(len(y_vals)), grid,
               levels=[-BAND], colors="white", linewidths=1.0,
               linestyles="solid")
    ax.plot(star[0], star[1], marker="*", markersize=11, color="#eda100",
            markeredgecolor=INK, markeredgewidth=0.6)
    ax.set_xticks(x_ticks[0])
    ax.set_xticklabels(x_ticks[1], fontsize=7)
    ax.set_yticks(y_ticks[0])
    ax.set_yticklabels(y_ticks[1], fontsize=7)
    ax.set_xlabel(x_label, fontsize=8)
    ax.set_ylabel(y_label, fontsize=8)
    return image


def main():
    c_ticks = None
    for dataset in DS_ORDER:
        paths = glob.glob(os.path.join("results", "surfaces", dataset + "__*.npz"))
        found = {os.path.basename(p).split("__")[1][:-4]: p for p in paths}
        models = [m for m in MODEL_ORDER if m in found]
        if not models:
            continue

        fig, axes = plt.subplots(len(models), 3, figsize=(10.5, 2.1 * len(models) + 0.6),
                                 squeeze=False)
        for row, model in enumerate(models):
            data = np.load(found[model])
            val = data["val"]                            # (alpha, k, c) mean val F1
            alpha_grid, k_grid, c_grid = data["alpha"], data["k"], data["c"]
            deficit = val - val.max()                    # 0 at the optimum, negative elsewhere
            ai, ki, ci = np.unravel_index(np.argmax(val), val.shape)

            # tick positions: every 10th c, every other alpha, every k
            c_idx = np.arange(0, len(c_grid), 10)
            c_ticks = (c_idx, ["%.1f" % c_grid[i] for i in c_idx])
            a_idx = np.arange(0, len(alpha_grid), 2)
            a_ticks = (a_idx, ["%g" % alpha_grid[i] for i in a_idx])
            k_ticks = (np.arange(len(k_grid)), ["%g" % k for k in k_grid])

            # slice 1: c (x) vs alpha (y) at k*
            image = draw_slice(axes[row, 0], deficit[:, ki, :], c_grid, alpha_grid,
                               "c", "alpha", c_ticks, a_ticks, (ci, ai))
            # slice 2: c (x) vs k (y) at alpha*
            draw_slice(axes[row, 1], deficit[ai, :, :], c_grid, k_grid,
                       "c", "k", c_ticks, k_ticks, (ci, ki))
            # slice 3: alpha (x) vs k (y) at c*
            draw_slice(axes[row, 2], deficit[:, :, ci].T, alpha_grid, k_grid,
                       "alpha", "k", a_ticks, k_ticks, (ai, ki))

            axes[row, 0].set_title("%s   k*=%g" % (MODEL_LABEL[model], k_grid[ki]),
                                   fontsize=8.5, color=INK, loc="left")
            axes[row, 1].set_title("alpha*=%g" % alpha_grid[ai], fontsize=8.5, color=INK, loc="left")
            axes[row, 2].set_title("c*=%.2f   val F1*=%.3f" % (c_grid[ci], val.max()),
                                   fontsize=8.5, color=INK, loc="left")

        fig.suptitle("%s: quantile_shrink validation objective, slices through the optimum"
                     % DS_LABEL[dataset], fontsize=10.5, color=INK)
        fig.tight_layout(rect=[0, 0, 0.92, 0.97])
        cbar_ax = fig.add_axes([0.935, 0.15, 0.012, 0.7])
        cbar = fig.colorbar(image, cax=cbar_ax)
        cbar.set_label("val F1 - optimum (white line = -%.2f)" % BAND, fontsize=8, color=SOFT)
        cbar.ax.tick_params(labelsize=7)
        out_path = os.path.join("results", "fig_surface_%s.png" % dataset)
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        print("wrote", out_path)


if __name__ == "__main__":
    main()
