"""
Task 5 predictor: writes the Section 8 output contract for a trained deep model,
WITH the Section 6 step-7 NN-propagation. It loads best.pt, runs a single
fixed-FPS pass over the val and test splits, scores the FPS points, then
propagates scores and logits to EVERY full-resolution t1 point by nearest
neighbour within the cube (on GPU), and saves one per-scene npz whose schema is
identical to the Task 4 ICP baseline so Task 7 calibration treats all six models
alike.

Why fixed FPS at predict time (Task 5 decision 5): the saved val npz must be
stable for Task 7, so we use a reproducible single FPS pass (default fps_seed=42)
for BOTH val and test. Training val FPS stays random per Section 7; that is a
separate loader and does not affect these saved predictions.

scores/logits convention (matches Task 4): logits are NN-propagated, then
scores = softmax(logits)[:,1] so logits and scores stay self-consistent (Task 7
platt/temperature act on logits, the others on scores).

NN-propagation is done in ORIGINAL metric coordinates (full_xyz1 and the scored
points full_xyz1[sel1] are both unnormalised). The cube normalisation is isotropic,
so nearest neighbours are identical in either frame; using original coords keeps
the distances physically meaningful. For a take-all cube (every point was scored)
the propagation is an exact identity.

Run (Kaggle GPU, after training):
    python src/predict.py --config configs/urb3dcd_v2_ld.yaml --model siamese_pointnet --num-workers 2
Local CPU plumbing check (after `train.py --smoke`):
    python src/predict.py --config configs/urb3dcd_v2_ld.yaml
"""

import os
import sys
import glob
import json
import argparse
import logging
import numpy as np
import yaml
import torch

# Same sibling-import setup as the other src/ scripts.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import build_model, is_fixed_n
from dataloader import build_cube_dataloader, loader_settings_from_config
from gpu_fps import pack_fps_batch
# Reuse the exact Task 1/2/3/4 PLY reader for the independent cross-check.
from inspect_dataset import read_cloud


def setup_logger(log_path, name):
    """Log to both the predict_<model>.log file and the console (Task 4 style)."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers = []                                  # avoid duplicate handlers on re-run
    fmt = logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S")
    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)
    return logger


def softmax_np(logits):
    """Numerically stable row-wise softmax for a [N, 2] float array."""
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def nn_propagate_cube(logits_input, sel1, full_xyz1, device):
    """
    Propagate the model's per-input-slot logits to every full-resolution t1 point.

    logits_input [Nin, 2] (on device): logits for the Nin model input slots. The
        first S = len(sel1) slots are the real scored points; any remaining slots
        are fixed-N padding and carry no meaning.
    sel1 [S] long: local indices of the scored points within full_xyz1.
    full_xyz1 [M, 3] float: ORIGINAL unnormalised t1 coords, full cube resolution.

    Returns (scores [M] f32, logits [M, 2] f32) with scores = softmax(logits)[:,1].
    For a take-all cube (S == M) every point maps to itself (exact identity).
    """
    full_xyz = full_xyz1.to(device)                       # [M, 3] all full-res t1 points
    sel = sel1.to(device)                                 # [S] scored-point indices
    num_scored = sel.shape[0]
    scored_logits = logits_input[:num_scored]             # [S, 2] real points carry the logits
    # CENTER the coordinates before computing distances. The saved coords are in the
    # dataset's projected CRS (values in the millions); torch.cdist's default expansion
    # ||x||^2 + ||y||^2 - 2<x,y> catastrophically cancels in float32 at that magnitude,
    # so the "nearest" scored point comes out essentially random and the propagation
    # scrambles the scores (AUC collapses to ~0.5). Distances are translation invariant,
    # so subtracting the cube centroid brings coords to ~tens of metres where float32 is
    # exact, without changing any nearest neighbour. Same lesson as the Task 4 ICP fix.
    centroid = full_xyz.mean(dim=0, keepdim=True)         # [1, 3] cube t1 centroid (any offset works)
    full_xyz_centered = full_xyz - centroid               # ~tens of metres, float32-safe
    scored_xyz = full_xyz_centered[sel]                   # [S, 3] scored points in the same frame
    # Nearest scored point for every full-res point, computed in ROW CHUNKS so the
    # [chunk, S] distance matrix stays small. Dense datasets (HKCD photogrammetry) reach
    # ~130k full-res t1 points per cube and S can be 8192, so a single [M, S] matrix would
    # be several GB and can OOM the GPU; chunking bounds peak memory to chunk_rows*S without
    # changing the result (argmin within a row block equals argmin over the whole matrix).
    # donot_use_mm avoids the cancellation-prone expansion as a second safeguard on top of
    # the centering. For LD/MS (small cubes) this is one chunk, so prior results are unchanged.
    chunk_rows = 20000                                    # 20000*8192 float32 ~= 0.6 GB per block
    nn_parts = []
    for start in range(0, full_xyz_centered.shape[0], chunk_rows):
        block = full_xyz_centered[start:start + chunk_rows]           # [c, 3] row block
        dist_block = torch.cdist(block, scored_xyz,
                                 compute_mode="donot_use_mm_for_euclid_dist")  # [c, S]
        nn_parts.append(dist_block.argmin(dim=1))         # [c] nearest scored point per row
    nn_index = torch.cat(nn_parts, dim=0)                 # [M] index of the nearest scored point
    prop_logits = scored_logits[nn_index]                 # [M, 2] copy that point's logits
    prop_scores = torch.softmax(prop_logits, dim=1)[:, 1]  # [M] P(change), self-consistent
    scores_np = prop_scores.cpu().numpy().astype(np.float32)
    logits_np = prop_logits.cpu().numpy().astype(np.float32)
    return scores_np, logits_np


def store_cube(store, scene_id, cube_id, cube_logits, sel1, full_xyz1, full_label1, device):
    """
    NN-propagate one cube's per-input-slot logits to full resolution and add the Section 8
    arrays to the store. Shared by the fixed-N and variable-N paths so the propagation and
    save format are identical for every model.
    """
    scores, logit2 = nn_propagate_cube(cube_logits, sel1, full_xyz1, device)
    num_full = scores.shape[0]
    arrays = (
        scores,                                            # [M] f32
        logit2,                                            # [M, 2] f32
        full_label1.numpy().astype(np.int64),             # [M] i64
        full_xyz1.numpy().astype(np.float32),             # [M, 3] f32 original XYZ
        np.full(num_full, cube_id, dtype=np.int32),       # [M] i32 cube id
    )
    store.setdefault(scene_id, {})[cube_id] = arrays


def predict_split(model, loader, device, fixed_n, gpu_fps=False, n_target=None, dl0=None):
    """
    Run one split end to end. Returns scene_id -> {cube_id -> (scores, logits,
    labels, coords, cube_id_array)}, accumulating one entry per scored cube.
    """
    store = {}
    model.eval()
    with torch.no_grad():
        for batch in loader:
            if gpu_fps:
                # GPU-FPS path: pack the raw batch ((optional grid subsample +) FPS on the model
                # device), with full-res arrays for NN-propagation. The grid subsample shrinks only
                # the scored input; full_xyz1/sel1 stay full-resolution so N and NN-prop are intact.
                batch = pack_fps_batch(batch, n_target, fixed_n, True, device, dl0=dl0)
            if fixed_n:
                # Fixed-N: stacked [B, N, 3] + mask; row i of logits maps to sel1[i].
                xyz0 = batch["xyz0"].to(device)
                xyz1 = batch["xyz1"].to(device)
                mask0 = batch["mask0"].to(device)
                mask1 = batch["mask1"].to(device)
                logits = model(xyz0, xyz1, mask0, mask1)  # [B, N, 2] on device
                for b in range(logits.shape[0]):
                    store_cube(store, batch["scene_id"][b], int(batch["cube_id"][b].item()),
                               logits[b], batch["sel1"][b], batch["full_xyz1"][b],
                               batch["full_label1"][b], device)
            else:
                # Variable-N (kpconv): per-cube forward; every input slot is a scored
                # point, in sel1 order, so the same per-cube NN-propagation applies.
                for i in range(len(batch["xyz1"])):
                    xyz0 = batch["xyz0"][i].to(device)    # [N0, 3] single cube
                    xyz1 = batch["xyz1"][i].to(device)    # [N1, 3]
                    cube_logits = model(xyz0, xyz1)       # [N1, 2] per-point t1 logits
                    store_cube(store, batch["scene_id"][i], int(batch["cube_id"][i].item()),
                               cube_logits, batch["sel1"][i], batch["full_xyz1"][i],
                               batch["full_label1"][i], device)
    return store


def assemble_scene(scene_store):
    """Concatenate a scene's cubes in cube order 0..K-1 into the Section 8 arrays."""
    cube_ids = sorted(scene_store.keys())
    # Cubes must be the contiguous 0..K-1 set from the cache (no gaps, no dupes).
    assert cube_ids == list(range(len(cube_ids))), "cube ids are not the contiguous 0..K-1 set"
    scores = np.concatenate([scene_store[c][0] for c in cube_ids]).astype(np.float32)
    logits = np.concatenate([scene_store[c][1] for c in cube_ids]).astype(np.float32)
    labels = np.concatenate([scene_store[c][2] for c in cube_ids]).astype(np.int64)
    coords = np.concatenate([scene_store[c][3] for c in cube_ids]).astype(np.float32)
    cube_id = np.concatenate([scene_store[c][4] for c in cube_ids]).astype(np.int32)
    return scores, logits, labels, coords, cube_id


def scene_f1_at_0p5(scores, labels):
    """Quick F1@0.5 of the change class for the log (sanity vs the ICP baseline)."""
    pred = scores > 0.5
    true_pos = int(np.sum(pred & (labels == 1)))
    false_pos = int(np.sum(pred & (labels == 0)))
    false_neg = int(np.sum((~pred) & (labels == 1)))
    precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0.0
    recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return f1, precision, recall


def auc_change(scores, labels):
    """
    ROC-AUC of the change class via Mann-Whitney U (average ranks for ties), numpy
    only. AUC is the threshold-free discrimination signal: F1@0.5 can look bad purely
    from miscalibration, but a low AUC means the model cannot rank change above
    no-change, which would cap EVERY calibration method (including two_moment). Logging
    it makes that failure mode obvious at a glance. Returns nan if a scene is one-class.
    """
    n_pos = int(np.sum(labels == 1))
    n_neg = int(len(labels) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")        # stable sort by score ascending
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)        # 1-based ranks
    # Average the ranks of tied scores so the AUC is exact under ties.
    s_sorted = scores[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    sum_pos_ranks = ranks[labels == 1].sum()
    return float((sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def independent_crosscheck(scene_rel, scene_id, raw_root, cache_root, npz_path,
                           ply_element, label_field):
    """
    Independent of the dataloader, rebuild the expected coords / labels / cube_id
    straight from the raw PLY pair + Task-2 cache and compare to the written npz.
    Also confirm scores == softmax(logits)[:,1]. Raises AssertionError on mismatch.
    """
    cache = np.load(os.path.join(cache_root, scene_id + ".npz"), allow_pickle=False)
    idx_t1 = cache["idx_t1"]                               # CSR t1 indices, cube order 0..K-1
    count_t1 = cache["count_t1"]                           # points per cube
    num_cubes = len(cache["cube_id"])
    scene_path = os.path.join(raw_root, *scene_rel.split("/"))
    xyz1, labels1 = read_cloud(os.path.join(scene_path, "pointCloud1.ply"), ply_element, label_field)
    if labels1 is None:
        labels1 = np.zeros(len(xyz1), dtype=np.int64)
    binary_labels1 = (labels1 > 0).astype(np.int64)       # collapse 0..6 -> {0,1}
    # Expected full-res arrays in cube order (idx_t1 is already CSR cube-ordered).
    expected_coords = xyz1[idx_t1].astype(np.float32)
    expected_labels = binary_labels1[idx_t1].astype(np.int64)
    expected_cube_id = np.repeat(np.arange(num_cubes, dtype=np.int32), count_t1)

    saved = np.load(npz_path, allow_pickle=True)
    assert np.array_equal(saved["coords"], expected_coords), "coords != xyz1[idx_t1]"
    assert np.array_equal(saved["labels"], expected_labels), "labels != bin1[idx_t1]"
    assert np.array_equal(saved["cube_id"], expected_cube_id), "cube_id != repeat(arange(K), count_t1)"
    # scores must equal softmax(logits)[:,1] within float tolerance.
    recomputed = softmax_np(saved["logits"])[:, 1]
    assert np.allclose(saved["scores"], recomputed, atol=1e-5), "scores != softmax(logits)[:,1]"
    return len(expected_coords)


def verify_schema(out_dir, scene_to_sum_count):
    """
    Section 8 schema-verification cell plus the per-scene N == sum(count_t1) check,
    run on every written file (identical to the Task 4 verifier).
    """
    files = sorted(glob.glob(os.path.join(out_dir, "*.npz")))
    assert len(files) > 0, "no .npz files written"
    for path in files:
        d = np.load(path, allow_pickle=True)
        for key in ["scores", "logits", "labels", "coords", "cube_id", "scene_id", "dataset"]:
            assert key in d.files, "missing key " + key + " in " + path
        assert d["scores"].dtype == np.float32 and d["scores"].ndim == 1
        assert d["logits"].dtype == np.float32 and d["logits"].shape[1] == 2
        assert d["labels"].dtype == np.int64
        assert d["coords"].dtype == np.float32 and d["coords"].shape[1] == 3
        assert d["cube_id"].dtype == np.int32
        n = len(d["scores"])
        assert n == len(d["labels"]) == len(d["coords"]) == len(d["cube_id"]) == len(d["logits"])
        scene_id = str(d["scene_id"])
        assert n == scene_to_sum_count[scene_id], (
            "N mismatch for " + scene_id + ": npz " + str(n)
            + " vs cache sum(count_t1) " + str(scene_to_sum_count[scene_id]))
    return len(files)


def main():
    parser = argparse.ArgumentParser(description="Predict + NN-propagate the Section 8 contract (Task 5).")
    parser.add_argument("--config", required=True, help="dataset YAML in configs/")
    parser.add_argument("--model", default="siamese_pointnet", help="Section 2 model name")
    parser.add_argument("--ckpt", default=None, help="checkpoint (default <out-root>/best.pt)")
    parser.add_argument("--out-root", default=None,
                        help="override output dir (default predictions/<dataset>/<model>)")
    parser.add_argument("--fps-seed", type=int, default=42,
                        help="fixed FPS seed for the reproducible predict pass (Section 7 test regime)")
    parser.add_argument("--batch-size", type=int, default=4, help="predict batch size")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="DataLoader workers (0 local/Windows, 2 on Kaggle Linux)")
    parser.add_argument("--gpu-fps", action="store_true",
                        help="run FPS+normalise+pad on the model device (match training; for MS/HKCD). "
                             "Default: CPU numpy FPS in the dataloader.")
    # Task 12 ablation knobs (must match the values used at train time for this ablation point).
    parser.add_argument("--fps-native", type=int, default=None,
                        help="override fps_native for the FPS sweep (Section 5.1); NEVER upsamples.")
    parser.add_argument("--cache-root", default=None,
                        help="override the cube cache dir for the cube-size sweep (Section 5.2).")
    args = parser.parse_args()

    # Task 6 adds the variable-N path: fixed_n picks the loader/collate and the per-cube
    # vs stacked forward convention. The NN-propagation that follows is identical for both.
    fixed_n = is_fixed_n(args.model)

    # Read paths and FPS native size from the dataset config.
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    dataset_name = cfg["dataset_name"]
    splits_file = cfg["splits_file"]
    settings = loader_settings_from_config(args.config)
    cache_root = settings["cache_root"]
    raw_root = settings["raw_root"]
    n_target = settings["n_target"]
    ply_element = settings["ply_element"]
    label_field = settings["label_field"]
    dl0 = settings["dl0"]              # grid-subsample voxel size (e.g. HKCD 1.0m), or None
    # Task 12 overrides (must match train.py for this ablation point); None on main runs.
    if args.fps_native is not None:
        n_target = args.fps_native
    if args.cache_root is not None:
        cache_root = args.cache_root

    out_root = args.out_root or os.path.join("predictions", dataset_name, args.model)
    ckpt_path = args.ckpt or os.path.join(out_root, "best.pt")
    os.makedirs(out_root, exist_ok=True)
    log = setup_logger(os.path.join(out_root, "predict_" + args.model + ".log"), "predict_" + args.model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Seed torch so the predict pass is reproducible. Most models are deterministic at
    # inference, but randla does Random Sampling INSIDE the network (RandLA-Net's signature
    # downsampling), so without a fixed seed the saved scores would vary run to run. The
    # dataloader's FPS uses its own per-cube crc32 seed, so this does not affect FPS selection.
    torch.manual_seed(args.fps_seed)
    torch.cuda.manual_seed_all(args.fps_seed)

    # Load the trained model.
    model = build_model(args.model).to(device)
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    log.info("predict: model=%s dataset=%s device=%s (fixed_n=%s)", args.model, dataset_name, device.type, fixed_n)
    log.info("effective fps_native=%d%s  cache_root=%s%s", n_target,
             "  [--fps-native OVERRIDE]" if args.fps_native is not None else "",
             cache_root, "  [--cache-root OVERRIDE]" if args.cache_root is not None else "")
    log.info("checkpoint=%s (epoch=%s val_f1=%.4f)", ckpt_path,
             checkpoint.get("epoch"), float(checkpoint.get("val_f1", float("nan"))))
    log.info("NN-propagation: per-cube nearest neighbour in centred metric space; scores=softmax(logits)[:,1]")
    log.info("predict FPS regime: FIXED seed=%d (reproducible single pass, val + test)", args.fps_seed)
    log.info("FPS device: %s", "model device (GPU-FPS, raw_mode loader)" if args.gpu_fps else "CPU numpy (dataloader)")
    log.info("grid subsample (dl0): %s", ("%.2f m before FPS (GPU-FPS only)" % dl0) if dl0 else "OFF (FPS over full cube)")
    log.info("output dir: %s", out_root)

    # Map scene_id -> relative path, and scene_id -> sum(count_t1) for verification.
    with open(splits_file) as f:
        splits = json.load(f)
    scene_rel_by_id = {}
    scene_to_sum_count = {}
    for split in ["val", "test"]:
        for scene_rel in splits[split]:
            scene_id = scene_rel.split("/")[-1]
            scene_rel_by_id[scene_id] = scene_rel
            cache = np.load(os.path.join(cache_root, scene_id + ".npz"), allow_pickle=False)
            scene_to_sum_count[scene_id] = int(cache["count_t1"].sum())

    log.info("")
    # Process val first (Task 7 fits calibration here) then test (applies it).
    for split in ["val", "test"]:
        loader = build_cube_dataloader(
            cache_root, raw_root, split, n_target,
            fps_seed=args.fps_seed, fixed_n=fixed_n, return_full_res=True,
            batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
            ply_element=ply_element, label_field=label_field, raw_mode=args.gpu_fps)
        store = predict_split(model, loader, device, fixed_n, gpu_fps=args.gpu_fps, n_target=n_target, dl0=dl0)

        for scene_id in sorted(store.keys()):
            scores, logits, labels, coords, cube_id = assemble_scene(store[scene_id])
            # Write the per-scene Section 8 npz (same keys/dtypes as Task 4).
            out_path = os.path.join(out_root, scene_id + ".npz")
            np.savez(
                out_path,
                scores=scores, logits=logits, labels=labels,
                coords=coords, cube_id=cube_id,
                scene_id=np.array(scene_id), dataset=np.array(dataset_name))

            # Independent cross-check straight from raw PLY + cache.
            n_checked = independent_crosscheck(
                scene_rel_by_id[scene_id], scene_id, raw_root, cache_root, out_path,
                ply_element, label_field)
            f1, precision, recall = scene_f1_at_0p5(scores, labels)
            auc = auc_change(scores, labels)             # threshold-free discrimination signal
            n_ok = "OK" if len(scores) == scene_to_sum_count[scene_id] else "MISMATCH"
            log.info("%-4s %-8s N=%d (cache %d %s) xcheck=%d chg=%.3f | AUC=%.3f F1@0.5=%.3f P=%.3f R=%.3f",
                     split, scene_id, len(scores), scene_to_sum_count[scene_id], n_ok,
                     n_checked, float((labels == 1).mean()), auc, f1, precision, recall)

    log.info("")
    log.info("Verifying Section 8 schema and N == sum(count_t1) on all written files...")
    n_files = verify_schema(out_root, scene_to_sum_count)
    log.info("Schema + N + cross-checks passed for %d scene files.", n_files)
    log.info("Done.")


if __name__ == "__main__":
    main()
