"""
GPU-side FPS for MS/HKCD training and prediction (Task 8 optimisation).

The numpy per-cube FPS in cube_sampling.py is fine for LD (fps 1024, small cubes) but on
MS (fps 4096, dense t1 cubes up to ~42k points) it costs ~1.3s per cube and runs EVERY
epoch (Section 7: train/val FPS random per batch), which makes the CPU the training
bottleneck. Moving FPS to the GPU one cube at a time was still too slow (~13 min/epoch:
a 4096-step Python loop per cube dominated by kernel-launch overhead). This module does it
BATCHED: all cubes in a batch share ONE 4096-step loop (vectorised over the batch), which
cuts the launch/Python overhead ~B-fold. It also moves the (frozen) normalisation and the
fixed-N padding onto the device, so the dataloader only slices raw points and the CPU is
freed (both T4s can train at once).

How it plugs in (opt-in, default OFF):
  - train.py / predict.py pass --gpu-fps. They build the loader in raw_mode (it returns the
    ragged ORIGINAL-coord cubes + bbox + a per-cube seed), and call pack_fps_batch() at the
    top of each batch loop.
  - pack_fps_batch runs batched FPS + normalisation + padding on `device`, then returns CPU
    tensors whose keys / shapes / dtypes are IDENTICAL to the CPU collate output. Every
    downstream line (model forward, masked loss, NN-propagation, the Section 8 save) is
    unchanged, and the existing `.to(device)` calls become harmless no-ops.
  - Without --gpu-fps the original CPU numpy path runs, so LD and prior results are untouched.

FPS regime preserved: a per-cube seed (the SAME crc32 stable_cube_seed the CPU path uses)
makes the predict pass reproducible; seed < 0 means the random train/val regime and uses the
global torch RNG. GPU FPS selects a DIFFERENT but equally valid subset than the numpy FPS;
that is fine - the contract only needs one valid reproducible pass and MS predictions are
generated fresh. The full-resolution coords / labels / cube_id saved by predict.py are
FPS-independent, so they are byte-identical either way; only which points get scored differs.

The normalisation MUST match cube_sampling.normalize_to_cube exactly (the frozen Task 3
transform): subtract the cube centre in XY and the ground (min Z), divide by half the cube
XY extent. The padding matches cube_sampling.pad_to_length / pad_labels.
"""

import torch


def _fps_padded(xyz_pad, counts, n_target, first_points):
    """
    Batched farthest point sampling over a padded [B, Mmax, 3] tensor. Every cube here has more
    than n_target real points (take-all cubes are handled in batched_fps). `counts[b]` is the
    number of REAL points in cube b (the first counts[b] rows of xyz_pad[b]); the rest are
    padding. `first_points[b]` is the (already drawn) seed index for cube b. Returns selected
    LOCAL indices [B, n_target]. Padded points are pinned to -inf nearest distance so they are
    never picked; since every cube has counts[b] > n_target, the n_target picks are all distinct
    real points (no duplicates).
    """
    device = xyz_pad.device
    batch_size, max_points, _ = xyz_pad.shape
    # Valid mask: True for the first counts[b] points of each cube.
    arange_m = torch.arange(max_points, device=device).unsqueeze(0)          # [1, Mmax]
    valid = arange_m < torch.tensor(counts, device=device).unsqueeze(1)      # [B, Mmax]
    # nearest-selected squared distance: +inf for real points, -inf for padding (never farthest).
    nearest_sq = torch.where(valid, torch.full_like(xyz_pad[:, :, 0], float("inf")),
                             torch.full_like(xyz_pad[:, :, 0], float("-inf")))   # [B, Mmax]
    selected = torch.empty(batch_size, n_target, dtype=torch.long, device=device)
    current = torch.as_tensor(first_points, dtype=torch.long, device=device)    # [B]
    selected[:, 0] = current
    batch_index = torch.arange(batch_size, device=device)                       # [B]
    # ONE loop for the whole batch; everything stays on device (no per-iteration sync).
    for i in range(1, n_target):
        current_xyz = xyz_pad[batch_index, current]            # [B, 3] the just-picked point per cube
        diff = xyz_pad - current_xyz.unsqueeze(1)              # [B, Mmax, 3]
        dist_sq = (diff * diff).sum(dim=2)                     # [B, Mmax] squared distances
        nearest_sq = torch.minimum(nearest_sq, dist_sq)       # padding stays -inf (min(-inf, .)=-inf)
        current = torch.argmax(nearest_sq, dim=1)             # [B] farthest remaining point per cube
        selected[:, i] = current
    return selected


def batched_fps(xyz_list, n_target, generators, device):
    """
    FPS for a batch of variable-length cubes. Returns a list of LOCAL index tensors (length
    min(M_i, n_target) per cube). Cubes with <= n_target points are take-all (arange, never
    upsample); the rest are sampled together in one batched loop (the big speedup). `generators[i]`
    is a torch.Generator (seeded/reproducible) or None (random regime); only the first point uses it.
    """
    selections = [None] * len(xyz_list)
    need = [i for i, xyz in enumerate(xyz_list) if xyz.shape[0] > n_target]   # cubes needing FPS
    # Take-all cubes: keep every point in order.
    for i, xyz in enumerate(xyz_list):
        if i not in need:
            selections[i] = torch.arange(xyz.shape[0], device=device)
    if need:
        counts = [int(xyz_list[i].shape[0]) for i in need]    # real point count per to-sample cube
        max_points = max(counts)
        xyz_pad = torch.zeros(len(need), max_points, 3, device=device)
        first_points = []
        for b, i in enumerate(need):
            xyz_pad[b, :counts[b]] = xyz_list[i]
            # First seed point per cube: reproducible if seeded, else from the global RNG.
            if generators[i] is not None:
                first_points.append(int(torch.randint(0, counts[b], (1,), generator=generators[i], device=device)[0]))
            else:
                first_points.append(int(torch.randint(0, counts[b], (1,), device=device)[0]))
        sel = _fps_padded(xyz_pad, counts, n_target, first_points)   # [len(need), n_target]
        for b, i in enumerate(need):
            selections[i] = sel[b]
    return selections


def normalize_to_cube_torch(xyz, bbox_min, bbox_max):
    """
    Frozen Task 3 normalisation, torch version - must match cube_sampling.normalize_to_cube:
    subtract the cube centre in XY and the ground (min Z), divide by half the cube XY extent.
    Shared isotropic transform, so the t0/t1 registration is preserved. xyz [M, 3] on device;
    bbox_min / bbox_max length-3 tensors on the same device. Returns float32 [M, 3].
    """
    centre = torch.stack([
        (bbox_min[0] + bbox_max[0]) * 0.5,               # cube centre X
        (bbox_min[1] + bbox_max[1]) * 0.5,               # cube centre Y
        bbox_min[2],                                     # ground level (min Z)
    ])
    half_extent = (bbox_max[0] - bbox_min[0]) * 0.5      # half the cube XY extent (one scalar)
    return (xyz - centre) / half_extent


def grid_subsample_torch(xyz, dl0, generator):
    """
    Voxel-grid subsampling: keep ONE point per dl0-metre cube voxel. Returns LOCAL indices
    into xyz (length G <= M), one representative (the lowest original index) per occupied voxel.

    This is the dense-data training strategy used by the Siamese-KPConv / Urb3DCD / PGN3DCD
    lineage (grid subsampling at a first-subsampling size dl0), adopted here so the expensive
    FPS runs over a few-thousand-point grid cloud instead of a ~15k-point photogrammetric cube
    (Task 9 / HKCD). It is O(M) hashing, far cheaper than FPS, and for a coarse enough dl0 the
    grid cloud falls below n_target so FPS is skipped entirely (take-all).

    A random per-draw voxel-origin offset (drawn from `generator`, so it is reproducible when
    seeded for predict and random per epoch for train) shifts which point represents each voxel,
    preserving the per-epoch input variation the two-moment method relies on. The returned indices
    address the ORIGINAL cube, so the caller still keeps the full-resolution cube for NN-propagation.
    """
    num_points = xyz.shape[0]
    if num_points == 0:
        return torch.arange(0, dtype=torch.long, device=xyz.device)
    # Random voxel-origin offset in [0, dl0): seeded -> reproducible, None -> fresh each epoch.
    if generator is not None:
        offset = torch.rand(3, generator=generator, device=xyz.device) * dl0
    else:
        offset = torch.rand(3, device=xyz.device) * dl0
    # Integer voxel coordinate of every point (shift to >=0 first so floor is well behaved).
    shifted = xyz - xyz.min(dim=0).values + offset                       # [M, 3] >= 0
    vox = torch.floor(shifted / dl0).long()                              # [M, 3] voxel index
    # Fold (vx, vy, vz) into one integer key; spans cover the cube extent so keys are unique/voxel.
    span = vox.max(dim=0).values + 1                                     # [3] per-axis voxel count
    key = (vox[:, 0] * span[1] + vox[:, 1]) * span[2] + vox[:, 2]        # [M] one key per voxel
    # One representative per voxel = the lowest original index falling in it.
    uniq, inverse = torch.unique(key, return_inverse=True)              # uniq [G], inverse [M]
    first_idx = torch.full((uniq.shape[0],), num_points, dtype=torch.long, device=xyz.device)
    arange_points = torch.arange(num_points, device=xyz.device)
    first_idx = first_idx.scatter_reduce(0, inverse, arange_points, reduce="amin", include_self=True)
    return first_idx                                                     # [G] original indices, one/voxel


def pack_fps_batch(raw_batch, n_target, fixed_n, return_full_res, device, dl0=None):
    """
    Turn a raw_mode batch (ragged original-coord cubes + bbox + per-cube seed, from
    dataloader.collate_cubes_raw) into the SAME structure the CPU collate functions produce,
    doing BATCHED FPS + normalisation + padding on `device`. The result is returned as CPU
    tensors so it is a drop-in replacement for the CPU collate output (train/predict then call
    .to(device) exactly as before).

    fixed_n True  -> xyz0/xyz1/mask0/mask1/label1 are stacked [B, n_target, ...] tensors.
    fixed_n False -> they are lists of per-cube tensors (variable native length).
    return_full_res True (predict) -> also full_xyz1 / full_label1 / sel1 lists for NN-prop.
    """
    batch_size = len(raw_batch["raw_xyz1"])
    xyz0_dev = [raw_batch["raw_xyz0"][i].to(device) for i in range(batch_size)]
    xyz1_dev = [raw_batch["raw_xyz1"][i].to(device) for i in range(batch_size)]

    # One generator per cube, SHARED across the two FPS calls (matches the CPU path's single
    # shared rng); None -> random regime. On MS t0 is take-all so only t1 actually draws.
    generators = []
    for i in range(batch_size):
        seed = int(raw_batch["fps_seed"][i])
        if seed >= 0:
            gen = torch.Generator(device=device)
            gen.manual_seed(seed)
            generators.append(gen)
        else:
            generators.append(None)

    # Optional grid (voxel) subsampling BEFORE FPS (dl0 set, e.g. HKCD). Each cube is reduced to
    # one point per dl0 voxel, so FPS runs over a few-thousand-point cloud (or is skipped when the
    # grid cloud already fits n_target). The grid index maps let us remap the FPS picks back to
    # ORIGINAL cube indices, so full_xyz1 / the saved labels / the Section 8 N stay full-resolution.
    if dl0 is not None and dl0 > 0:
        gmap0 = [grid_subsample_torch(xyz0_dev[i], dl0, generators[i]) for i in range(batch_size)]
        gmap1 = [grid_subsample_torch(xyz1_dev[i], dl0, generators[i]) for i in range(batch_size)]
        red0 = [xyz0_dev[i][gmap0[i]] for i in range(batch_size)]   # reduced t0 clouds
        red1 = [xyz1_dev[i][gmap1[i]] for i in range(batch_size)]   # reduced t1 clouds
        # FPS over the reduced clouds gives indices INTO the reduced cloud...
        fps0 = batched_fps(red0, n_target, generators, device)
        fps1 = batched_fps(red1, n_target, generators, device)
        # ...remap to original-cube indices via the grid maps (so sel addresses the full cube).
        sel0_list = [gmap0[i][fps0[i]] for i in range(batch_size)]
        sel1_list = [gmap1[i][fps1[i]] for i in range(batch_size)]
    else:
        # No grid subsampling (LD/MS): FPS directly over the full cube, exactly as before.
        sel0_list = batched_fps(xyz0_dev, n_target, generators, device)
        sel1_list = batched_fps(xyz1_dev, n_target, generators, device)

    in0_list, in1_list = [], []
    mask0_list, mask1_list, lab_list = [], [], []
    full_xyz1_list, full_label1_list, sel1_cpu_list = [], [], []
    for i in range(batch_size):
        bbox_min = raw_batch["cube_bbox_min"][i].to(device)
        bbox_max = raw_batch["cube_bbox_max"][i].to(device)
        in0 = normalize_to_cube_torch(xyz0_dev[i][sel0_list[i]], bbox_min, bbox_max)   # [m0, 3]
        in1 = normalize_to_cube_torch(xyz1_dev[i][sel1_list[i]], bbox_min, bbox_max)   # [m1, 3]
        lab = raw_batch["raw_label1"][i].to(device)[sel1_list[i]]                      # [m1]

        if fixed_n:
            # Pad take-all cubes up to n_target + mask (same as cube_sampling.pad_to_length).
            m0 = in0.shape[0]
            m1 = in1.shape[0]
            padded0 = torch.zeros(n_target, 3, device=device); padded0[:m0] = in0
            padded1 = torch.zeros(n_target, 3, device=device); padded1[:m1] = in1
            mask0 = torch.zeros(n_target, dtype=torch.bool, device=device); mask0[:m0] = True
            mask1 = torch.zeros(n_target, dtype=torch.bool, device=device); mask1[:m1] = True
            padded_lab = torch.zeros(n_target, dtype=torch.long, device=device); padded_lab[:m1] = lab
            in0_list.append(padded0); in1_list.append(padded1)
            mask0_list.append(mask0); mask1_list.append(mask1); lab_list.append(padded_lab)
        else:
            in0_list.append(in0); in1_list.append(in1)
            mask0_list.append(torch.ones(in0.shape[0], dtype=torch.bool, device=device))
            mask1_list.append(torch.ones(in1.shape[0], dtype=torch.bool, device=device))
            lab_list.append(lab)

        if return_full_res:
            # predict.py needs full-resolution t1 (ORIGINAL coords) + the scored-point indices.
            full_xyz1_list.append(raw_batch["raw_xyz1"][i])      # CPU float (the whole cube)
            full_label1_list.append(raw_batch["raw_label1"][i])  # CPU long
            sel1_cpu_list.append(sel1_list[i].to("cpu"))

    # Assemble exactly like the CPU collate output, on CPU (downstream .to(device) works).
    out = {"scene_id": raw_batch["scene_id"], "cube_id": raw_batch["cube_id"]}
    if fixed_n:
        out["xyz0"] = torch.stack(in0_list).cpu()
        out["xyz1"] = torch.stack(in1_list).cpu()
        out["mask0"] = torch.stack(mask0_list).cpu()
        out["mask1"] = torch.stack(mask1_list).cpu()
        out["label1"] = torch.stack(lab_list).cpu()
    else:
        out["xyz0"] = [t.cpu() for t in in0_list]
        out["xyz1"] = [t.cpu() for t in in1_list]
        out["mask0"] = [t.cpu() for t in mask0_list]
        out["mask1"] = [t.cpu() for t in mask1_list]
        out["label1"] = [t.cpu() for t in lab_list]
    if return_full_res:
        out["full_xyz1"] = full_xyz1_list
        out["full_label1"] = full_label1_list
        out["sel1"] = sel1_cpu_list
    return out
