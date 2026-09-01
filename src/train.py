"""
Task 5 trainer for the deep change-detection models. This is the FIRST deep model,
so it sets the pattern Tasks 6 and 8 reuse: the architecture is chosen by --model
and built through models.build_model, while the training RECIPE is identical for
every model (Section 7 parity, architecture is the only variable):

    60 epochs, Adam lr 1e-3, CosineAnnealingLR(T_max=60), batch 4 (drop to 2 if OOM
    and log it), weighted CE with per-dataset class weights, seed 42, frozen splits,
    train + val FPS random per draw (essential augmentation for the two-moment method).

What it does:
  1. Compute inverse-frequency class weights from the TRAIN cubes, read straight
     from the Task-2 cache counts (no PLY read needed): w_c = N / (2 * n_c).
  2. Train --model on the train split; each epoch evaluate val F1 of the change class.
  3. Keep the checkpoint with the best val F1, scored at that epoch's F1-optimal
     threshold (a quick val sweep, so checkpoint quality is not pinned to tau=0.5).
  4. Save best.pt {state_dict, epoch, val_f1, val_tau, model_name, class_weights}
     and train_<model>.log.

Prediction and the Section 8 npz contract (with NN-propagation) live in
src/predict.py, which loads best.pt.

Run (Kaggle GPU, full recipe):
    python src/train.py --config configs/urb3dcd_v2_ld.yaml --model siamese_pointnet --num-workers 2
Local CPU plumbing check (1 epoch, a few batches, throwaway checkpoint):
    python src/train.py --config configs/urb3dcd_v2_ld.yaml --smoke
"""

import os
import sys
import json
import time
import random
import argparse
import logging
import numpy as np
import yaml
import torch
import torch.nn as nn

# Same sys.path trick as the other src/ scripts so sibling imports resolve.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import build_model, is_fixed_n
from dataloader import build_cube_dataloader, loader_settings_from_config
from gpu_fps import pack_fps_batch


def setup_logger(log_path, name):
    """Log to both the train_<model>.log file and the console (same style as Task 4)."""
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


def set_seed(seed):
    """Seed python, numpy and torch. cudnn nondeterminism is left ON for speed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_class_weights(cache_root, splits_file):
    """
    Inverse-frequency class weights from the TRAIN split, summed over the cache
    per-cube counts (no PLY read). Returns (weights [w_unchanged, w_changed] float32,
    (n_unchanged, n_changed, n_total)). Weights use the sklearn-balanced convention
    w_c = N / (num_classes * n_c), so the frequency-weighted mean weight is 1.
    """
    with open(splits_file) as f:
        splits = json.load(f)
    n_unchanged = 0
    n_changed = 0
    for scene_rel in splits["train"]:
        scene_id = scene_rel.split("/")[-1]               # leaf folder is the scene id
        cache = np.load(os.path.join(cache_root, scene_id + ".npz"), allow_pickle=False)
        # n_unchanged / n_changed are per-cube full-resolution t1 counts; sum them.
        n_unchanged += int(cache["n_unchanged"].sum())
        n_changed += int(cache["n_changed"].sum())
    n_total = n_unchanged + n_changed
    # Balanced inverse-frequency weights (2 classes: 0 unchanged, 1 changed).
    w_unchanged = n_total / (2.0 * n_unchanged)
    w_changed = n_total / (2.0 * n_changed)
    weights = np.array([w_unchanged, w_changed], dtype=np.float32)
    return weights, (n_unchanged, n_changed, n_total)


def f1_change(scores, labels, tau):
    """F1 of the change class (label 1) at threshold tau; scores/labels are 1D numpy."""
    pred = scores > tau                                   # boolean change prediction
    true_pos = int(np.sum(pred & (labels == 1)))
    false_pos = int(np.sum(pred & (labels == 0)))
    false_neg = int(np.sum((~pred) & (labels == 1)))
    precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0.0
    recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def variable_n_logits_labels(model, batch, device):
    """
    Forward a variable-N (kpconv) batch ONE CUBE AT A TIME and concatenate the per-point
    t1 logits and labels over the batch's cubes. The collate hands lists (one ragged
    cloud per cube); every point is real, so there is no mask. Returns
    (logits [P, 2], labels [P]) over all t1 points in the batch.
    """
    logits_list = []
    labels_list = []
    for i in range(len(batch["xyz1"])):                  # one entry per cube in the batch
        xyz0 = batch["xyz0"][i].to(device)               # [N0, 3] single cube, no padding
        xyz1 = batch["xyz1"][i].to(device)               # [N1, 3]
        logits_i = model(xyz0, xyz1)                     # [N1, 2] per-point t1 logits
        logits_list.append(logits_i)
        labels_list.append(batch["label1"][i].to(device))   # [N1] binary t1 labels
    return torch.cat(logits_list, dim=0), torch.cat(labels_list, dim=0)


def train_one_epoch(model, loader, optimizer, criterion, device, fixed_n,
                    gpu_fps=False, n_target=None, limit_batches=None, dl0=None):
    """One pass over the train split. Returns the mean masked loss over batches."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    for batch_index, batch in enumerate(loader):
        if limit_batches is not None and batch_index >= limit_batches:
            break                                         # smoke test only
        if gpu_fps:
            # GPU-FPS path: the loader handed a raw batch; do (optional grid subsample +) FPS +
            # normalise + pad on the model device now. Result matches the CPU collate output.
            batch = pack_fps_batch(batch, n_target, fixed_n, False, device, dl0=dl0)
        if fixed_n:
            # Fixed-N path (siamese_pointnet/pointnet2/siamgcn): stacked [B, N, 3] + mask.
            xyz0 = batch["xyz0"].to(device)
            xyz1 = batch["xyz1"].to(device)
            mask0 = batch["mask0"].to(device)
            mask1 = batch["mask1"].to(device)
            labels = batch["label1"].to(device)          # [B, N] binary t1 labels
            logits = model(xyz0, xyz1, mask0, mask1)     # [B, N, 2] per-point logits
            # Per-point class-weighted CE, then keep REAL t1 points only via the mask.
            loss_per_point = criterion(logits.reshape(-1, 2), labels.reshape(-1))   # [B*N]
            mask_flat = mask1.reshape(-1).float()        # 1.0 real, 0.0 padded
            loss = (loss_per_point * mask_flat).sum() / mask_flat.sum().clamp(min=1.0)
        else:
            # Variable-N path (siamese_kpconv): per-cube forward, every point is real.
            logits, labels = variable_n_logits_labels(model, batch, device)   # [P, 2], [P]
            loss_per_point = criterion(logits, labels)   # [P] class-weighted CE
            loss = loss_per_point.mean()                 # mean over all real points
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
        num_batches += 1
    return total_loss / max(num_batches, 1)


def evaluate(model, loader, device, fixed_n, gpu_fps=False, n_target=None, limit_batches=None, dl0=None):
    """
    Evaluate on the val split. Collects scores/labels over REAL t1 points and returns
    (f1_at_0p5, best_f1, best_tau) where best_f1 is the max F1 over a coarse tau sweep.
    """
    model.eval()
    all_scores = []
    all_labels = []
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if limit_batches is not None and batch_index >= limit_batches:
                break                                     # smoke test only
            if gpu_fps:
                batch = pack_fps_batch(batch, n_target, fixed_n, False, device, dl0=dl0)
            if fixed_n:
                xyz0 = batch["xyz0"].to(device)
                xyz1 = batch["xyz1"].to(device)
                mask0 = batch["mask0"].to(device)
                mask1 = batch["mask1"].to(device)
                logits = model(xyz0, xyz1, mask0, mask1)  # [B, N, 2]
                probs = torch.softmax(logits, dim=2)[:, :, 1]  # [B, N] P(change)
                real_dev = mask1                          # [B, N] bool on device
                # Select real points; the cpu mask selects the SAME positions in label1,
                # so scores and labels stay aligned (both flatten in [B,N] row-major order).
                all_scores.append(probs[real_dev].cpu().numpy())
                real_cpu = batch["mask1"]                 # [B, N] bool on cpu
                all_labels.append(batch["label1"][real_cpu].numpy())
            else:
                # Variable-N: per-cube forward, all points real (no mask to apply).
                logits, labels = variable_n_logits_labels(model, batch, device)   # [P, 2], [P]
                probs = torch.softmax(logits, dim=1)[:, 1]   # [P] P(change)
                all_scores.append(probs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
    scores = np.concatenate(all_scores)
    labels = np.concatenate(all_labels)
    f1_05 = f1_change(scores, labels, 0.5)
    # Coarse threshold sweep to score the checkpoint at its F1-optimal operating point.
    best_f1 = 0.0
    best_tau = 0.5
    for tau in np.arange(0.05, 0.96, 0.05):
        f1 = f1_change(scores, labels, float(tau))
        if f1 > best_f1:
            best_f1 = f1
            best_tau = float(tau)
    return f1_05, best_f1, best_tau


def main():
    parser = argparse.ArgumentParser(description="Train a deep change-detection model (Task 5).")
    parser.add_argument("--config", required=True, help="dataset YAML in configs/")
    parser.add_argument("--model", default="siamese_pointnet", help="Section 2 model name")
    parser.add_argument("--epochs", type=int, default=None,
                        help="total training epochs / cosine horizon (Section 7 = 60; default 60, or 1 in --smoke)")
    parser.add_argument("--batch-size", type=int, default=4, help="batch size (drop to 2 if OOM)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate (Section 7 = 1e-3)")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="DataLoader workers (0 local/Windows, 2 on Kaggle Linux)")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed (Section 7 = 42)")
    parser.add_argument("--out-root", default=None,
                        help="override output dir (default predictions/<dataset>/<model>)")
    parser.add_argument("--smoke", action="store_true",
                        help="1 epoch + a few batches for a local plumbing check (NOT the recipe)")
    parser.add_argument("--gpu-fps", action="store_true",
                        help="run FPS+normalise+pad on the model device (frees the CPU; for MS/HKCD "
                             "where dense-cube FPS is the CPU bottleneck). Default: CPU numpy FPS.")
    parser.add_argument("--resume", action="store_true",
                        help="continue from <out-root>/last.pt if it exists (multi-commit training on "
                             "time-limited platforms like Kaggle). Harmless on the first run (starts fresh).")
    parser.add_argument("--max-hours", type=float, default=None,
                        help="stop cleanly after this many wall-clock hours (checked after each epoch), "
                             "saving last.pt and exiting 0 so the run persists. Re-run with --resume to "
                             "continue. Set below the platform session limit (e.g. 7.5 on Kaggle). "
                             "Default: no limit (run all epochs).")
    # Task 12 ablation knobs. Both default to None (use the config value) so main runs are unchanged.
    parser.add_argument("--fps-native", type=int, default=None,
                        help="override fps_native for the FPS sweep (Section 5.1). NEVER upsamples: a value "
                             "above a cube's point count just takes all + pads, same as the main runs.")
    parser.add_argument("--cache-root", default=None,
                        help="override the cube cache dir for the cube-size sweep (Section 5.2); pair with a "
                             "per-size cache built by preprocess_cubes.py --cube-size S --cache-root ...")
    args = parser.parse_args()

    # Task 6 adds the variable-N path: fixed_n picks the loader/collate and the
    # forward convention (stacked+mask vs per-cube). build_model rejects unknown names.
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
    # Task 12 overrides: --fps-native swaps the FPS target (FPS sweep), --cache-root swaps the
    # cube cache (cube-size sweep). Left None on main runs -> config values, nothing changes.
    if args.fps_native is not None:
        n_target = args.fps_native
    if args.cache_root is not None:
        cache_root = args.cache_root

    out_root = args.out_root or os.path.join("predictions", dataset_name, args.model)
    os.makedirs(out_root, exist_ok=True)
    log = setup_logger(os.path.join(out_root, "train_" + args.model + ".log"), "train_" + args.model)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Resolve the epoch horizon: explicit --epochs wins; otherwise 1 for a smoke plumbing
    # check, 60 for the real Section 7 recipe. This horizon is also the cosine T_max, so it
    # stays fixed at 60 across resumed commits (only --max-hours decides where a commit stops).
    if args.epochs is not None:
        epochs = args.epochs
    else:
        epochs = 1 if args.smoke else 60
    limit_train = 3 if args.smoke else None
    limit_val = 3 if args.smoke else None

    # Inverse-frequency class weights from the TRAIN cubes.
    class_weights, (n_unchanged, n_changed, n_total) = compute_class_weights(cache_root, splits_file)

    log.info("training: model=%s dataset=%s (fixed_n=%s)", args.model, dataset_name, fixed_n)
    log.info("effective fps_native=%d%s  cache_root=%s%s", n_target,
             "  [--fps-native OVERRIDE, Section 5.1 sweep]" if args.fps_native is not None else "",
             cache_root, "  [--cache-root OVERRIDE, Section 5.2 sweep]" if args.cache_root is not None else "")
    log.info("device=%s | epochs=%d batch=%d lr=%g cosine T_max=%d seed=%d workers=%d%s",
             device.type, epochs, args.batch_size, args.lr, epochs, args.seed, args.num_workers,
             "  [SMOKE]" if args.smoke else "")
    log.info("recipe: Adam + CosineAnnealingLR, weighted CE, train/val FPS random (Section 7)")
    log.info("cudnn nondeterminism left ON for speed (train FPS is random by design anyway)")
    log.info("FPS device: %s", "model device (GPU-FPS, raw_mode loaders)" if args.gpu_fps else "CPU numpy (dataloader)")
    log.info("grid subsample (dl0): %s", ("%.2f m before FPS (GPU-FPS only)" % dl0) if dl0 else "OFF (FPS over full cube)")
    log.info("class counts (train, full-res t1): unchanged=%d changed=%d total=%d change_ratio=%.4f",
             n_unchanged, n_changed, n_total, n_changed / n_total)
    log.info("class weights (balanced inverse freq, w=N/(2*n_c)): unchanged=%.6f changed=%.6f",
             class_weights[0], class_weights[1])
    if args.batch_size != 4:
        log.info("NOTE: batch size %d differs from the Section 7 default of 4 (document the reason)",
                 args.batch_size)

    # Train loader: random FPS (fps_seed=None) + shuffle (the default for that regime).
    train_loader = build_cube_dataloader(
        cache_root, raw_root, "train", n_target,
        fps_seed=None, fixed_n=fixed_n, return_full_res=False,
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        ply_element=ply_element, label_field=label_field, raw_mode=args.gpu_fps)
    # Val loader: FPS random per Section 7, but keep cube order stable (shuffle=False).
    val_loader = build_cube_dataloader(
        cache_root, raw_root, "val", n_target,
        fps_seed=None, fixed_n=fixed_n, return_full_res=False,
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        ply_element=ply_element, label_field=label_field, raw_mode=args.gpu_fps)
    log.info("loaders: train cubes=%d val cubes=%d", len(train_loader.dataset), len(val_loader.dataset))

    # Build the model and the Section 7 optimiser / schedule / loss.
    model = build_model(args.model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
    # reduction='none' so we can mask out padded points before averaging the loss.
    criterion = nn.CrossEntropyLoss(weight=weight_tensor, reduction="none")

    best_f1 = -1.0
    best_epoch = -1
    best_tau = 0.5
    # best.pt = the best-val-F1 snapshot (what predict.py loads). last.pt = the FULL training
    # state (model+optimizer+scheduler+epoch+best) so --resume can continue a run that was split
    # across several time-limited commits. Both paths are set before the loop.
    ckpt_path = os.path.join(out_root, "best.pt")
    last_path = os.path.join(out_root, "last.pt")
    start_epoch = 1

    # Resume: pick up exactly where the previous commit stopped. Harmless if last.pt is absent
    # (first commit) -> starts fresh. The cosine T_max stays = epochs, and scheduler.load_state_dict
    # restores its internal last_epoch, so the LR schedule continues correctly across commits.
    if args.resume and os.path.exists(last_path):
        checkpoint = torch.load(last_path, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_f1 = float(checkpoint["best_f1"])
        best_epoch = int(checkpoint["best_epoch"])
        best_tau = float(checkpoint["best_tau"])
        log.info("RESUMED from %s: completed epoch %d -> continue at %d/%d (best val_f1=%.4f @ep%d)",
                 last_path, checkpoint["epoch"], start_epoch, epochs, best_f1, best_epoch)
    elif args.resume:
        log.info("--resume set but no %s yet; starting fresh at epoch 1", last_path)

    run_start = time.time()        # wall clock for --max-hours (measures THIS commit only, not resumed epochs)
    stopped_early = False
    epoch = start_epoch - 1        # so the summary is correct even if the loop body never runs
    log.info("")
    log.info("starting training... epochs %d..%d%s", start_epoch, epochs,
             ("" if args.max_hours is None else "  (stop after %.2f h, then --resume)" % args.max_hours))
    for epoch in range(start_epoch, epochs + 1):
        epoch_start = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, fixed_n,
                                     gpu_fps=args.gpu_fps, n_target=n_target, limit_batches=limit_train, dl0=dl0)
        f1_05, val_f1, val_tau = evaluate(model, val_loader, device, fixed_n,
                                          gpu_fps=args.gpu_fps, n_target=n_target, limit_batches=limit_val, dl0=dl0)
        scheduler.step()                                  # cosine step once per epoch
        current_lr = optimizer.param_groups[0]["lr"]
        seconds = time.time() - epoch_start
        log.info("epoch %2d/%d | train_loss=%.4f | val F1@0.5=%.4f | val bestF1=%.4f @tau=%.2f | "
                 "lr=%.2e | %.1fs",
                 epoch, epochs, train_loss, f1_05, val_f1, val_tau, current_lr, seconds)
        # Keep the checkpoint with the best val F1 (at its own F1-optimal threshold).
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch
            best_tau = val_tau
            # Snapshot weights to cpu so the saved checkpoint is device-independent.
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            # Save IMMEDIATELY on every improvement so any stop still leaves the best-so-far checkpoint.
            torch.save({
                "state_dict": best_state,
                "epoch": best_epoch,
                "val_f1": best_f1,
                "val_tau": best_tau,                      # informational; Task 7 re-derives tau
                "model_name": args.model,
                "class_weights": class_weights.tolist(),
            }, ckpt_path)
        # Save the FULL training state every epoch so a resumed commit continues seamlessly.
        torch.save({
            "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,                               # last COMPLETED epoch
            "best_f1": best_f1, "best_epoch": best_epoch, "best_tau": best_tau,
            "model_name": args.model,
            "class_weights": class_weights.tolist(),
            "total_epochs": epochs,
        }, last_path)
        # Clean stop before the platform session limit: state is already saved, so just leave the
        # loop, let the commit finish (exit 0) and persist, then re-run with --resume to continue.
        if args.max_hours is not None and (time.time() - run_start) / 3600.0 >= args.max_hours:
            stopped_early = True
            log.info("reached --max-hours=%.2f after epoch %d/%d; stopping cleanly (re-run with --resume)",
                     args.max_hours, epoch, epochs)
            break

    log.info("")
    if start_epoch > epochs:
        # Resumed a run that was already finished; nothing to train.
        log.info("ALREADY COMPLETE: %d/%d epochs done (best val_f1=%.4f @ep%d). Run predict.py.",
                 epochs, epochs, best_f1, best_epoch)
    elif stopped_early:
        log.info("PARTIAL: trained through epoch %d/%d this commit. best val_f1=%.4f @ep%d. "
                 "Re-run with --resume to continue.", epoch, epochs, best_f1, best_epoch)
    else:
        log.info("COMPLETE: %d/%d epochs. best epoch=%d val_f1=%.4f (tau=%.2f) -> %s",
                 epochs, epochs, best_epoch, best_f1, best_tau, ckpt_path)
        log.info("Next: python src/predict.py --config %s --model %s", args.config, args.model)


if __name__ == "__main__":
    main()
