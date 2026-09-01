# Task 1 — Urb3DCD-V2 inventory (LD + MS)

Source: IEEE-DataPort `IEEE_Dataset_V2_Lid05_MS.zip` (~452 MB, login-gated).
One zip bundles both sub-datasets: `1-Lidar05` -> `urb3dcd_v2_ld`,
`5-MultiSensor` -> `urb3dcd_v2_ms`. Raw data lives under `data/raw/<dataset>/`.

PLY format (verified from the author's loader): element name `params`, fields
`x, y, z, label_ch`. Scene = a folder with `pointCloud0.ply` (t0) and
`pointCloud1.ply` (t1). Change labels are on t1, 7 classes
(0 unchanged, 1 newlyBuilt, 2 deconstructed, 3 newVegetation, 4 vegetationGrowUp,
5 vegetationRemoved, 6 mobileObjects), collapsed to binary for this project.

## Scenes (identical Lyon scenes in both sub-datasets)

| split | scenes | scene names |
|-------|--------|-------------|
| train | 10 | LyonN, LyonN1, LyonN2, LyonN3, LyonN4, LyonN5, LyonN9, LyonN11, LyonN12, LyonN14 |
| val   | 2  | Lyon, Lyon1 |
| test  | 3  | LyonS, LyonS3, LyonS4 |

Frozen as `data/splits/<dataset>/splits.json` (the official release split).

## Cube counts at 50 x 50 x full Z, min 256 pts in both clouds

| dataset | split | occupied | kept | dropped |
|---------|-------|----------|------|---------|
| LD | train | 1546 | 1394 | 152 |
| LD | val   | 463  | 423  | 40  |
| LD | test  | 852  | 765  | 87  |
| LD | TOTAL | 2861 | 2582 | 279 |
| MS | train | 1555 | 1406 | 149 |
| MS | val   | 464  | 429  | 35  |
| MS | test  | 853  | 768  | 85  |
| MS | TOTAL | 2872 | 2603 | 269 |

Scenes span ~1 km; ~10% of occupied cubes are dropped (sparse edges). The sparse
t0 cloud is the binding constraint for the drop rule in both datasets.

## Per-cube point density of kept cubes (median, points per cube)

| dataset | cloud | p10 | median | p90 | max |
|---------|-------|-----|--------|-----|-----|
| LD | t0 | 748   | 1145  | 1608  | 2141  |
| LD | t1 | 743   | 1149  | 1643  | 2126  |
| MS | t0 | 717   | 1146  | 1604  | 2139  |
| MS | t1 | 14389 | 22976 | 32531 | 42260 |

## FPS implication (decision for Tasks 2-3, do not silently upsample)

Frozen FPS-native targets are 1024 (LD) and 4096 (MS).
- LD @ 1024: median cube (1145) downsamples cleanly, but the sparsest ~30-40% of
  cubes (p10 ~748) fall below 1024 and would be upsampled to reach it.
- MS @ 4096: dense t1 (median 22976) downsamples ~5.6x, but sparse t0 (median 1146)
  would be upsampled ~3.6x. The two MS dates use different sensors.

This collides with the Section 5.1 "NEVER upsample" rule. Tasks 2/3 must choose a
policy: per-cloud FPS targets, sample-with-replacement, or take-all-points + pad.

Per-scene detail: `results/urb3dcd_v2_ld_inventory.csv`, `results/urb3dcd_v2_ms_inventory.csv`.
