"""
Freeze the IndoorCD train/val/test split (Task 11). IndoorCD ships with no
official split (just Data/<scene>/ and Label/), so we create ONE frozen split
here and never resplit it (CLAUDE.md Section 7).

IndoorCD layout: each source scene <scene> has a reference scan <scene>-1.pcd
(no label) and one or more modified scans <scene>-N.pcd, each with a change
label Label/<scene>-N.json. We treat every (scan1, scanN) pair as one
bi-temporal sample, with sample id "<scene>-N".

The split is by SOURCE SCENE, not by pair: all pairs of a scene go to the same
split. This stops the shared reference scan (scan1) from leaking between splits
(scan1 is the t0 of every pair in that scene).

Output: data/splits/indoorcd/splits.json, same structure as the other datasets
(folder_map + train/val/test lists of "<Folder>/<sample_id>"). The converter
(src/convert_indoorcd.py) reads this file and writes each pair under its folder.

Run:
    python src/freeze_indoorcd_split.py
"""

import os
import glob
import json
import numpy as np

RAW = "data/raw/IndoorCD"
LABEL_DIR = os.path.join(RAW, "Label")
OUT_FILE = "data/splits/indoorcd/splits.json"

# Split fractions over source scenes. 70/10/20 gives a healthy val (for fitting
# the two-moment (c, lambda)) and a large test (many scenes for the per-scene
# scope-limit statistics), unlike HKCD's single val scene.
TRAIN_FRAC = 0.70
VAL_FRAC = 0.10
SEED = 42  # Section 7 fixed seed; the split is deterministic given this seed.

FOLDER_MAP = {"train": "Train", "val": "Val", "test": "Test"}


def main():
    # Enumerate every (scene, modified scan) pair from the label files.
    label_files = sorted(glob.glob(os.path.join(LABEL_DIR, "*.json")))
    pairs = []  # sample ids like "001-2"
    for lf in label_files:
        sample_id = os.path.basename(lf)[:-5]  # strip ".json"
        pairs.append(sample_id)

    # Group pair ids by their source scene (the part before the dash).
    pairs_by_scene = {}
    for sample_id in pairs:
        scene = sample_id.split("-")[0]
        pairs_by_scene.setdefault(scene, []).append(sample_id)

    scenes = sorted(pairs_by_scene.keys())
    print("source scenes:", len(scenes), "| pairs:", len(pairs))

    # Deterministic shuffle of the scene list, then cut by fraction.
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(scenes))
    shuffled = [scenes[i] for i in order]
    n_train = int(round(TRAIN_FRAC * len(shuffled)))
    n_val = int(round(VAL_FRAC * len(shuffled)))
    train_scenes = set(shuffled[:n_train])
    val_scenes = set(shuffled[n_train:n_train + n_val])
    test_scenes = set(shuffled[n_train + n_val:])

    # Build the split lists of "<Folder>/<sample_id>", pairs sorted for stability.
    split_lists = {"train": [], "val": [], "test": []}
    for sample_id in sorted(pairs):
        scene = sample_id.split("-")[0]
        if scene in train_scenes:
            split = "train"
        elif scene in val_scenes:
            split = "val"
        else:
            split = "test"
        split_lists[split].append(FOLDER_MAP[split] + "/" + sample_id)

    note = (
        "Frozen split for IndoorCD (Task 11 scope-limit study). No official split "
        "exists, so this is one frozen split by SOURCE SCENE (all pairs of a scene "
        "share a split to stop the shared reference scan from leaking), "
        + str(int(TRAIN_FRAC * 100)) + "/" + str(int(VAL_FRAC * 100)) + "/"
        + str(int((1 - TRAIN_FRAC - VAL_FRAC) * 100)) + " by scene, seed "
        + str(SEED) + ". A sample is a (scan1, scanN) bi-temporal pair, id "
        "'<scene>-N'. Never resplit (CLAUDE.md Section 7).")

    out = {
        "dataset": "indoorcd",
        "note": note,
        "folder_map": FOLDER_MAP,
        "train": split_lists["train"],
        "val": split_lists["val"],
        "test": split_lists["test"],
    }

    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    with open(OUT_FILE, "w") as f:
        json.dump(out, f, indent=2)

    print("scenes  train/val/test:", len(train_scenes), len(val_scenes), len(test_scenes))
    print("pairs   train/val/test:",
          len(split_lists["train"]), len(split_lists["val"]), len(split_lists["test"]))
    print("written:", OUT_FILE)


if __name__ == "__main__":
    main()
