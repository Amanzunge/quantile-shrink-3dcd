"""
Task 3 cube-pair dataloader. Serves bi-temporal cube pairs to the deep Siamese
models, applying the three FPS regimes from Section 7:
  train  FPS random per batch   (fps_seed=None -> fresh random subset each draw)
  val    FPS random             (fps_seed=None, same as train, used while training)
  test   FPS fixed seed         (fps_seed=int -> reproducible single pass per cube)

The Task-2 cache holds only point INDICES (CSR), not coordinates, so this module
reads each scene's raw PLY pair once at construction (via the Task-1 read_cloud),
keeps the clouds in memory, and recovers a cube's points by slicing the CSR
indices. The heavy maths (FPS, normalisation, padding) lives in the torch-free
src/cube_sampling.py so it stays unit-testable on a CPU-only box.

What one sample carries:
  xyz0, xyz1   normalized model input for the t0 and t1 cube clouds
  mask0, mask1 True for real points, False for fixed-N padding
  label1       binary change label per t1 input point (0 unchanged, 1 changed)
  scene_id, cube_id   identity for writing the Section 8 npz later
And, only when return_full_res=True (predict.py needs this for NN-propagation):
  full_xyz1    ORIGINAL UNNORMALISED t1 cube coords, full resolution
  full_label1  binary labels for every full-resolution t1 point
  sel1         local indices of the FPS-scored points within full_xyz1
  cube_bbox_min, cube_bbox_max   the cube box, for reference

This module imports torch; the torch-free core does not. Run training/inference
where torch exists (Kaggle, or the global interpreter locally for a smoke test).
"""

import os
import json
import zlib
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Reuse the exact Task-1 PLY reader so coordinates and labels match the cache.
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inspect_dataset import read_cloud
from cube_sampling import (
    farthest_point_sampling,
    normalize_to_cube,
    pad_to_length,
    pad_labels,
)


def stable_cube_seed(base_seed, scene_id, cube_id):
    """
    Build a reproducible per-cube seed from a base seed and the cube identity.
    crc32 of "scene:cube" is stable across runs and machines (unlike Python's
    salted hash), so the test/predict FPS pass is byte-for-byte repeatable.
    """
    key = (str(scene_id) + ":" + str(int(cube_id))).encode("utf-8")
    return (int(base_seed) + zlib.crc32(key)) % (2 ** 32)


class CubePairDataset(Dataset):
    """
    One item per kept cube in the chosen split. Reads the split's scene clouds
    once up front; __getitem__ slices a cube, runs the FPS regime, normalises, and
    (for fixed-N models) pads to n_target with a mask.
    """

    def __init__(self, cache_root, raw_root, split, n_target,
                 fps_seed=None, fixed_n=True, return_full_res=False,
                 ply_element="params", label_field="label_ch", raw_mode=False):
        # n_target is the model native size (FPS target), e.g. 1024 LD / 4096 MS.
        self.n_target = int(n_target)
        # fps_seed None -> random regime (train/val); int -> fixed regime (test).
        self.fps_seed = fps_seed
        # fixed_n True -> pad to n_target + mask (pointnet/pointnet2/siamgcn);
        # False -> variable-N ragged output (kpconv grid / randla large-N).
        self.fixed_n = bool(fixed_n)
        # return_full_res True -> also expose full-resolution t1 for NN-propagation.
        self.return_full_res = bool(return_full_res)
        # raw_mode True -> skip FPS/normalise/pad and return the ragged ORIGINAL-coord cube
        # plus bbox and a per-cube seed; gpu_fps.pack_fps_batch does FPS on the model device
        # (the --gpu-fps path for MS/HKCD). The CPU numpy path (raw_mode False) is the default.
        self.raw_mode = bool(raw_mode)

        # Read the dataset manifest and keep only the requested split's scenes.
        manifest_file = os.path.join(cache_root, "manifest.json")
        with open(manifest_file) as f:
            manifest = json.load(f)
        scene_rows = [r for r in manifest["scenes"] if r["split"] == split]
        if len(scene_rows) == 0:
            raise ValueError("no scenes for split '" + split + "' in " + manifest_file)

        # Per-scene in-memory stores, keyed by scene_id.
        self.clouds = {}   # scene_id -> (xyz0, xyz1, binary_labels1)
        self.caches = {}   # scene_id -> the cube CSR + bbox arrays
        # Flat list of (scene_id, cube_id); one entry per kept cube in the split.
        self.index = []

        for row in scene_rows:
            scene_id = row["scene_id"]
            # Load the Task-2 cube cache for this scene (indices + boxes only).
            cache_path = os.path.join(cache_root, scene_id + ".npz")
            cache = np.load(cache_path, allow_pickle=False)
            self.caches[scene_id] = {
                "idx_t0": cache["idx_t0"],
                "offset_t0": cache["offset_t0"],
                "idx_t1": cache["idx_t1"],
                "offset_t1": cache["offset_t1"],
                "cube_bbox_min": cache["cube_bbox_min"],
                "cube_bbox_max": cache["cube_bbox_max"],
            }

            # Read the raw cloud pair once; the cache indices address these arrays.
            scene_path = os.path.join(raw_root, *row["scene_rel"].split("/"))
            xyz0, _ = read_cloud(
                os.path.join(scene_path, "pointCloud0.ply"), ply_element, label_field)
            xyz1, labels1 = read_cloud(
                os.path.join(scene_path, "pointCloud1.ply"), ply_element, label_field)
            if labels1 is None:
                labels1 = np.zeros(len(xyz1), dtype=np.int64)
            # Collapse the 0..6 source labels to binary change (any of 1..6 -> 1).
            binary_labels1 = (labels1 > 0).astype(np.int64)
            self.clouds[scene_id] = (xyz0, xyz1, binary_labels1)

            # One flat entry per cube; cube_id is the cache's canonical 0..K-1.
            num_cubes = len(cache["cube_id"])
            for cube_id in range(num_cubes):
                self.index.append((scene_id, cube_id))

    def __len__(self):
        return len(self.index)

    def _cube_rng(self, scene_id, cube_id):
        """Random Generator for one cube: fresh if random regime, seeded if fixed."""
        if self.fps_seed is None:
            # Fresh OS entropy each call -> a different FPS subset every epoch.
            return np.random.default_rng()
        # Deterministic per-cube seed -> the test pass is reproducible.
        return np.random.default_rng(stable_cube_seed(self.fps_seed, scene_id, cube_id))

    def __getitem__(self, i):
        scene_id, cube_id = self.index[i]
        xyz0, xyz1, binary_labels1 = self.clouds[scene_id]
        cache = self.caches[scene_id]

        # Slice this cube's raw point indices out of the CSR layout.
        s0, e0 = cache["offset_t0"][cube_id], cache["offset_t0"][cube_id + 1]
        s1, e1 = cache["offset_t1"][cube_id], cache["offset_t1"][cube_id + 1]
        cube_xyz0 = xyz0[cache["idx_t0"][s0:e0]]            # [M0, 3] raw t0 points
        cube_xyz1 = xyz1[cache["idx_t1"][s1:e1]]            # [M1, 3] raw t1 points
        cube_lab1 = binary_labels1[cache["idx_t1"][s1:e1]]  # [M1] binary labels

        bbox_min = cache["cube_bbox_min"][cube_id]
        bbox_max = cache["cube_bbox_max"][cube_id]

        if self.raw_mode:
            # GPU-FPS path: hand back the ragged ORIGINAL-coord cube + bbox + a per-cube seed;
            # FPS / normalise / pad happen later in gpu_fps.pack_fps_batch on the model device.
            # seed = the same crc32 per-cube seed the CPU path uses (reproducible predict), or
            # -1 for the random train/val regime (pack uses the global torch RNG).
            seed = stable_cube_seed(self.fps_seed, scene_id, cube_id) if self.fps_seed is not None else -1
            return {
                "raw_xyz0": cube_xyz0.astype(np.float32),
                "raw_xyz1": cube_xyz1.astype(np.float32),
                "raw_label1": cube_lab1.astype(np.int64),
                "cube_bbox_min": np.asarray(bbox_min, dtype=np.float32),
                "cube_bbox_max": np.asarray(bbox_max, dtype=np.float32),
                "scene_id": scene_id,
                "cube_id": int(cube_id),
                "fps_seed": int(seed),
            }

        # FPS each cloud with the regime's rng (shared rng, drawn sequentially).
        rng = self._cube_rng(scene_id, cube_id)
        sel0 = farthest_point_sampling(cube_xyz0, self.n_target, rng)
        sel1 = farthest_point_sampling(cube_xyz1, self.n_target, rng)

        # Normalise the selected points with the shared per-cube transform.
        in0 = normalize_to_cube(cube_xyz0[sel0], bbox_min, bbox_max)
        in1 = normalize_to_cube(cube_xyz1[sel1], bbox_min, bbox_max)
        lab_sel = cube_lab1[sel1]

        if self.fixed_n:
            # Pad take-all cubes up to n_target and carry a real-point mask.
            in0, mask0 = pad_to_length(in0, self.n_target)
            in1, mask1 = pad_to_length(in1, self.n_target)
            lab_out = pad_labels(lab_sel, self.n_target)
        else:
            # Variable-N models keep the native length; every point is real.
            mask0 = np.ones(in0.shape[0], dtype=bool)
            mask1 = np.ones(in1.shape[0], dtype=bool)
            lab_out = lab_sel

        sample = {
            "xyz0": in0,
            "xyz1": in1,
            "mask0": mask0,
            "mask1": mask1,
            "label1": lab_out,
            "scene_id": scene_id,
            "cube_id": int(cube_id),
        }

        if self.return_full_res:
            # Everything predict.py needs to NN-propagate FPS scores to full res.
            # full_xyz1 stays ORIGINAL UNNORMALISED per the Section 8 contract.
            sample["full_xyz1"] = cube_xyz1.astype(np.float32)
            sample["full_label1"] = cube_lab1.astype(np.int64)
            sample["sel1"] = sel1.astype(np.int64)
            sample["cube_bbox_min"] = np.asarray(bbox_min, dtype=np.float32)
            sample["cube_bbox_max"] = np.asarray(bbox_max, dtype=np.float32)

        return sample


def collate_cubes_fixed_n(batch):
    """
    Collate fixed-N samples by stacking. The per-point arrays all have length
    n_target, so they stack into dense [B, n_target, ...] tensors. Variable-length
    full-resolution fields (if present) are kept as python lists, one per sample.
    """
    out = {
        "xyz0": torch.from_numpy(np.stack([b["xyz0"] for b in batch])).float(),
        "xyz1": torch.from_numpy(np.stack([b["xyz1"] for b in batch])).float(),
        "mask0": torch.from_numpy(np.stack([b["mask0"] for b in batch])).bool(),
        "mask1": torch.from_numpy(np.stack([b["mask1"] for b in batch])).bool(),
        "label1": torch.from_numpy(np.stack([b["label1"] for b in batch])).long(),
        "scene_id": [b["scene_id"] for b in batch],
        "cube_id": torch.tensor([b["cube_id"] for b in batch], dtype=torch.long),
    }
    if "full_xyz1" in batch[0]:
        # Full-res cubes differ in length, so they cannot stack; keep as lists.
        out["full_xyz1"] = [torch.from_numpy(b["full_xyz1"]).float() for b in batch]
        out["full_label1"] = [torch.from_numpy(b["full_label1"]).long() for b in batch]
        out["sel1"] = [torch.from_numpy(b["sel1"]).long() for b in batch]
        out["cube_bbox_min"] = [torch.from_numpy(b["cube_bbox_min"]).float() for b in batch]
        out["cube_bbox_max"] = [torch.from_numpy(b["cube_bbox_max"]).float() for b in batch]
    return out


def collate_cubes_variable_n(batch):
    """
    Collate variable-N samples (kpconv grid / randla) as lists of per-cube tensors,
    since the clouds have different lengths and cannot be stacked. cube_id stacks
    into one tensor; everything point-shaped stays a list.
    """
    out = {
        "xyz0": [torch.from_numpy(b["xyz0"]).float() for b in batch],
        "xyz1": [torch.from_numpy(b["xyz1"]).float() for b in batch],
        "mask0": [torch.from_numpy(b["mask0"]).bool() for b in batch],
        "mask1": [torch.from_numpy(b["mask1"]).bool() for b in batch],
        "label1": [torch.from_numpy(b["label1"]).long() for b in batch],
        "scene_id": [b["scene_id"] for b in batch],
        "cube_id": torch.tensor([b["cube_id"] for b in batch], dtype=torch.long),
    }
    if "full_xyz1" in batch[0]:
        out["full_xyz1"] = [torch.from_numpy(b["full_xyz1"]).float() for b in batch]
        out["full_label1"] = [torch.from_numpy(b["full_label1"]).long() for b in batch]
        out["sel1"] = [torch.from_numpy(b["sel1"]).long() for b in batch]
        out["cube_bbox_min"] = [torch.from_numpy(b["cube_bbox_min"]).float() for b in batch]
        out["cube_bbox_max"] = [torch.from_numpy(b["cube_bbox_max"]).float() for b in batch]
    return out


def collate_cubes_raw(batch):
    """
    Collate for the GPU-FPS path (raw_mode): return the ragged ORIGINAL-coord cubes, bbox, and
    per-cube seed as python lists. FPS / normalise / pad are deferred to gpu_fps.pack_fps_batch,
    which runs them on the model device and emits the same structure the other collates produce.
    """
    out = {
        "raw_xyz0": [torch.from_numpy(b["raw_xyz0"]).float() for b in batch],
        "raw_xyz1": [torch.from_numpy(b["raw_xyz1"]).float() for b in batch],
        "raw_label1": [torch.from_numpy(b["raw_label1"]).long() for b in batch],
        "cube_bbox_min": [torch.from_numpy(b["cube_bbox_min"]).float() for b in batch],
        "cube_bbox_max": [torch.from_numpy(b["cube_bbox_max"]).float() for b in batch],
        "scene_id": [b["scene_id"] for b in batch],
        "cube_id": torch.tensor([b["cube_id"] for b in batch], dtype=torch.long),
        "fps_seed": [int(b["fps_seed"]) for b in batch],
    }
    return out


def build_cube_dataloader(cache_root, raw_root, split, n_target,
                          fps_seed=None, fixed_n=True, return_full_res=False,
                          batch_size=4, shuffle=None, num_workers=0,
                          ply_element="params", label_field="label_ch", raw_mode=False):
    """
    Construct a CubePairDataset and wrap it in a DataLoader with the right collate.
    shuffle defaults to the random regime (train): shuffle when fps_seed is None,
    keep order for the fixed regime (predict). Pass shuffle explicitly to override
    (e.g. shuffle=False for a val pass during training).

    raw_mode True selects the GPU-FPS path: the dataset returns ragged raw cubes and the
    collate is collate_cubes_raw; the caller runs gpu_fps.pack_fps_batch per batch. fixed_n and
    return_full_res are then handled in pack_fps_batch, not the dataset.
    """
    dataset = CubePairDataset(
        cache_root, raw_root, split, n_target,
        fps_seed=fps_seed, fixed_n=fixed_n, return_full_res=return_full_res,
        ply_element=ply_element, label_field=label_field, raw_mode=raw_mode)

    if shuffle is None:
        # Random FPS regime -> training -> shuffle; fixed regime -> ordered pass.
        shuffle = fps_seed is None

    if raw_mode:
        collate = collate_cubes_raw
    elif fixed_n:
        collate = collate_cubes_fixed_n
    else:
        collate = collate_cubes_variable_n
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, collate_fn=collate)
    return loader


def loader_settings_from_config(config_path):
    """
    Pull the paths and FPS native size out of a dataset YAML so callers do not
    hardcode them. Returns a dict with cache_root, raw_root, n_target, and the
    PLY element/label field. Cube size and min points are already baked into the
    cache, so they are not needed here.
    """
    import yaml
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    # grid_subsample (dl0, metres) is OPTIONAL: when set (e.g. HKCD), the GPU-FPS path voxel-
    # subsamples each cube to one point per dl0 voxel before FPS. Absent/None -> no subsampling
    # (LD/MS behave exactly as before). Only the --gpu-fps path uses it.
    grid_subsample = cfg.get("grid_subsample", None)
    return {
        "cache_root": cfg["cache_root"],
        "raw_root": cfg["raw_root"],
        "n_target": int(cfg["fps_native"]),
        "ply_element": cfg.get("ply_element", "params"),
        "label_field": cfg.get("label_field", "label_ch"),
        "dl0": float(grid_subsample) if grid_subsample is not None else None,
    }
