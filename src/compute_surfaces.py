"""
compute_surfaces.py - Cache the full quantile_shrink objective surface.

For every model x dataset pair, scores every (alpha, k, c) candidate of the
calibrate_v2.fit_quantile_shrink grid on val (the objective that is maximised)
and on test (what that candidate would have deployed). Uses the same
grid_surfaces walk as src/sensitivity.py, but keeps the whole surface on disk
so plotting never has to recompute it.

Torch-free, local CPU:  python src/compute_surfaces.py
Writes results/surfaces/<dataset>__<model>.npz with keys
  val, test   (n_alpha, n_k, n_c) mean per-scene F1
  alpha, k, c the grid axes
"""

import os
import json
import numpy as np
import yaml

import calibrate_v2 as cv
from sensitivity import CONFIGS, grid_surfaces

OUT_DIR = os.path.join("results", "surfaces")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for config in CONFIGS:
        cfg = yaml.safe_load(open(config))
        dataset = cfg["dataset_name"]

        # scene name -> split, from the frozen splits file
        splits = json.load(open(cfg["splits_file"]))
        split_of = {}
        for split in ["train", "val", "test"]:
            for rel in splits.get(split, []):
                split_of[rel.split("/")[-1]] = split

        pred_root = os.path.join("predictions", dataset)
        models = sorted(m for m in os.listdir(pred_root)
                        if os.path.isdir(os.path.join(pred_root, m)))

        for model in models:
            out_path = os.path.join(OUT_DIR, "%s__%s.npz" % (dataset, model))
            if os.path.exists(out_path):
                print("skip (cached)", out_path)
                continue
            val, test = cv.load_split_scores(os.path.join(pred_root, model), split_of)
            val_pre = [cv.precompute_scene(rec) for rec in val]
            test_pre = [cv.precompute_scene(rec) for rec in test]

            val_surface = grid_surfaces(val_pre)
            test_surface = grid_surfaces(test_pre)
            np.savez(out_path, val=val_surface, test=test_surface,
                     alpha=cv.ALPHA_GRID, k=cv.K_GRID, c=cv.C_GRID)
            print("wrote", out_path, "val max %.4f" % val_surface.max())


if __name__ == "__main__":
    main()
