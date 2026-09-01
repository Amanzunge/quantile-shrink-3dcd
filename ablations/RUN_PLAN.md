# Task 12 ablation run plan (deep GPU runs)

ABLATION_EPOCHS = **60** (full Section-7 parity, CLAUDE 5.4). Every job = train.py then predict.py with the SAME flags, `--gpu-fps --resume --epochs 60`, writing to `<DRIVE>/<out_root_rel>`. Use `colab/ablation_colab.py` (one job per cell run, resume across sessions). 66 deep jobs total: 28 FPS + 38 cube. The main anchors (cube 50/1.5, FPS 1024/4096) and all ICP cube points are ALREADY done locally -- do NOT rerun them.

## Drive uploads (build with kaggle_upload/make_zips.py, copy to MyDrive/threshold/)

- `code.zip` (REBUILT this chat -- has --fps-native/--cache-root). REQUIRED for every job.
- FPS sweep reuses the MAIN caches: `urb3dcd_v2_ld_{raw,meta}.zip`, `urb3dcd_v2_ms_{raw,meta}.zip` (already built).
- Cube sweep needs the per-size meta zips (built this chat): `<ds>_cube<size>_meta.zip` for LD/MS/HKCD {25,75} and IndoorCD {1.0,2.0}, PLUS each dataset's raw zip (`urb3dcd_v2_ld_raw.zip`, `urb3dcd_v2_ms_raw.zip`, build `hkcd_raw.zip` via `make_zips.py raw configs/hkcd.yaml`, `indoorcd_ply.zip` already built).

## FPS sweep jobs (28)

| dataset | model | fps | batch | out_root_rel |
|---|---|---|---|---|
| urb3dcd_v2_ld | siamese_pointnet | 256 | 4 | ablations/urb3dcd_v2_ld/fps/256/siamese_pointnet |
| urb3dcd_v2_ld | siamese_pointnet2 | 256 | 4 | ablations/urb3dcd_v2_ld/fps/256/siamese_pointnet2 |
| urb3dcd_v2_ld | siamese_kpconv | 256 | 2 | ablations/urb3dcd_v2_ld/fps/256/siamese_kpconv |
| urb3dcd_v2_ld | siamgcn | 256 | 2 | ablations/urb3dcd_v2_ld/fps/256/siamgcn |
| urb3dcd_v2_ld | siamese_pointnet | 512 | 4 | ablations/urb3dcd_v2_ld/fps/512/siamese_pointnet |
| urb3dcd_v2_ld | siamese_pointnet2 | 512 | 4 | ablations/urb3dcd_v2_ld/fps/512/siamese_pointnet2 |
| urb3dcd_v2_ld | siamese_kpconv | 512 | 2 | ablations/urb3dcd_v2_ld/fps/512/siamese_kpconv |
| urb3dcd_v2_ld | siamgcn | 512 | 2 | ablations/urb3dcd_v2_ld/fps/512/siamgcn |
| urb3dcd_v2_ms | siamese_pointnet | 256 | 4 | ablations/urb3dcd_v2_ms/fps/256/siamese_pointnet |
| urb3dcd_v2_ms | siamese_pointnet2 | 256 | 4 | ablations/urb3dcd_v2_ms/fps/256/siamese_pointnet2 |
| urb3dcd_v2_ms | siamese_kpconv | 256 | 2 | ablations/urb3dcd_v2_ms/fps/256/siamese_kpconv |
| urb3dcd_v2_ms | siamgcn | 256 | 2 | ablations/urb3dcd_v2_ms/fps/256/siamgcn |
| urb3dcd_v2_ms | randla | 256 | 2 | ablations/urb3dcd_v2_ms/fps/256/randla |
| urb3dcd_v2_ms | siamese_pointnet | 512 | 4 | ablations/urb3dcd_v2_ms/fps/512/siamese_pointnet |
| urb3dcd_v2_ms | siamese_pointnet2 | 512 | 4 | ablations/urb3dcd_v2_ms/fps/512/siamese_pointnet2 |
| urb3dcd_v2_ms | siamese_kpconv | 512 | 2 | ablations/urb3dcd_v2_ms/fps/512/siamese_kpconv |
| urb3dcd_v2_ms | siamgcn | 512 | 2 | ablations/urb3dcd_v2_ms/fps/512/siamgcn |
| urb3dcd_v2_ms | randla | 512 | 2 | ablations/urb3dcd_v2_ms/fps/512/randla |
| urb3dcd_v2_ms | siamese_pointnet | 1024 | 4 | ablations/urb3dcd_v2_ms/fps/1024/siamese_pointnet |
| urb3dcd_v2_ms | siamese_pointnet2 | 1024 | 4 | ablations/urb3dcd_v2_ms/fps/1024/siamese_pointnet2 |
| urb3dcd_v2_ms | siamese_kpconv | 1024 | 2 | ablations/urb3dcd_v2_ms/fps/1024/siamese_kpconv |
| urb3dcd_v2_ms | siamgcn | 1024 | 2 | ablations/urb3dcd_v2_ms/fps/1024/siamgcn |
| urb3dcd_v2_ms | randla | 1024 | 2 | ablations/urb3dcd_v2_ms/fps/1024/randla |
| urb3dcd_v2_ms | siamese_pointnet | 2048 | 4 | ablations/urb3dcd_v2_ms/fps/2048/siamese_pointnet |
| urb3dcd_v2_ms | siamese_pointnet2 | 2048 | 4 | ablations/urb3dcd_v2_ms/fps/2048/siamese_pointnet2 |
| urb3dcd_v2_ms | siamese_kpconv | 2048 | 2 | ablations/urb3dcd_v2_ms/fps/2048/siamese_kpconv |
| urb3dcd_v2_ms | siamgcn | 2048 | 2 | ablations/urb3dcd_v2_ms/fps/2048/siamgcn |
| urb3dcd_v2_ms | randla | 2048 | 2 | ablations/urb3dcd_v2_ms/fps/2048/randla |

## Cube-size sweep jobs (38)

| dataset | model | cube cache | batch | out_root_rel |
|---|---|---|---|---|
| urb3dcd_v2_ld | siamese_pointnet | urb3dcd_v2_ld_cube25 | 4 | ablations/urb3dcd_v2_ld/cube/25/siamese_pointnet |
| urb3dcd_v2_ld | siamese_pointnet2 | urb3dcd_v2_ld_cube25 | 4 | ablations/urb3dcd_v2_ld/cube/25/siamese_pointnet2 |
| urb3dcd_v2_ld | siamese_kpconv | urb3dcd_v2_ld_cube25 | 2 | ablations/urb3dcd_v2_ld/cube/25/siamese_kpconv |
| urb3dcd_v2_ld | siamgcn | urb3dcd_v2_ld_cube25 | 2 | ablations/urb3dcd_v2_ld/cube/25/siamgcn |
| urb3dcd_v2_ld | siamese_pointnet | urb3dcd_v2_ld_cube75 | 4 | ablations/urb3dcd_v2_ld/cube/75/siamese_pointnet |
| urb3dcd_v2_ld | siamese_pointnet2 | urb3dcd_v2_ld_cube75 | 4 | ablations/urb3dcd_v2_ld/cube/75/siamese_pointnet2 |
| urb3dcd_v2_ld | siamese_kpconv | urb3dcd_v2_ld_cube75 | 2 | ablations/urb3dcd_v2_ld/cube/75/siamese_kpconv |
| urb3dcd_v2_ld | siamgcn | urb3dcd_v2_ld_cube75 | 2 | ablations/urb3dcd_v2_ld/cube/75/siamgcn |
| urb3dcd_v2_ms | siamese_pointnet | urb3dcd_v2_ms_cube25 | 4 | ablations/urb3dcd_v2_ms/cube/25/siamese_pointnet |
| urb3dcd_v2_ms | siamese_pointnet2 | urb3dcd_v2_ms_cube25 | 4 | ablations/urb3dcd_v2_ms/cube/25/siamese_pointnet2 |
| urb3dcd_v2_ms | siamese_kpconv | urb3dcd_v2_ms_cube25 | 2 | ablations/urb3dcd_v2_ms/cube/25/siamese_kpconv |
| urb3dcd_v2_ms | siamgcn | urb3dcd_v2_ms_cube25 | 2 | ablations/urb3dcd_v2_ms/cube/25/siamgcn |
| urb3dcd_v2_ms | randla | urb3dcd_v2_ms_cube25 | 2 | ablations/urb3dcd_v2_ms/cube/25/randla |
| urb3dcd_v2_ms | siamese_pointnet | urb3dcd_v2_ms_cube75 | 4 | ablations/urb3dcd_v2_ms/cube/75/siamese_pointnet |
| urb3dcd_v2_ms | siamese_pointnet2 | urb3dcd_v2_ms_cube75 | 4 | ablations/urb3dcd_v2_ms/cube/75/siamese_pointnet2 |
| urb3dcd_v2_ms | siamese_kpconv | urb3dcd_v2_ms_cube75 | 2 | ablations/urb3dcd_v2_ms/cube/75/siamese_kpconv |
| urb3dcd_v2_ms | siamgcn | urb3dcd_v2_ms_cube75 | 2 | ablations/urb3dcd_v2_ms/cube/75/siamgcn |
| urb3dcd_v2_ms | randla | urb3dcd_v2_ms_cube75 | 2 | ablations/urb3dcd_v2_ms/cube/75/randla |
| hkcd | siamese_pointnet | hkcd_cube25 | 4 | ablations/hkcd/cube/25/siamese_pointnet |
| hkcd | siamese_pointnet2 | hkcd_cube25 | 4 | ablations/hkcd/cube/25/siamese_pointnet2 |
| hkcd | siamese_kpconv | hkcd_cube25 | 2 | ablations/hkcd/cube/25/siamese_kpconv |
| hkcd | siamgcn | hkcd_cube25 | 2 | ablations/hkcd/cube/25/siamgcn |
| hkcd | randla | hkcd_cube25 | 2 | ablations/hkcd/cube/25/randla |
| hkcd | siamese_pointnet | hkcd_cube75 | 4 | ablations/hkcd/cube/75/siamese_pointnet |
| hkcd | siamese_pointnet2 | hkcd_cube75 | 4 | ablations/hkcd/cube/75/siamese_pointnet2 |
| hkcd | siamese_kpconv | hkcd_cube75 | 2 | ablations/hkcd/cube/75/siamese_kpconv |
| hkcd | siamgcn | hkcd_cube75 | 2 | ablations/hkcd/cube/75/siamgcn |
| hkcd | randla | hkcd_cube75 | 2 | ablations/hkcd/cube/75/randla |
| indoorcd | siamese_pointnet | indoorcd_cube1.0 | 4 | ablations/indoorcd/cube/1.0/siamese_pointnet |
| indoorcd | siamese_pointnet2 | indoorcd_cube1.0 | 4 | ablations/indoorcd/cube/1.0/siamese_pointnet2 |
| indoorcd | siamese_kpconv | indoorcd_cube1.0 | 2 | ablations/indoorcd/cube/1.0/siamese_kpconv |
| indoorcd | siamgcn | indoorcd_cube1.0 | 2 | ablations/indoorcd/cube/1.0/siamgcn |
| indoorcd | randla | indoorcd_cube1.0 | 2 | ablations/indoorcd/cube/1.0/randla |
| indoorcd | siamese_pointnet | indoorcd_cube2.0 | 4 | ablations/indoorcd/cube/2.0/siamese_pointnet |
| indoorcd | siamese_pointnet2 | indoorcd_cube2.0 | 4 | ablations/indoorcd/cube/2.0/siamese_pointnet2 |
| indoorcd | siamese_kpconv | indoorcd_cube2.0 | 2 | ablations/indoorcd/cube/2.0/siamese_kpconv |
| indoorcd | siamgcn | indoorcd_cube2.0 | 2 | ablations/indoorcd/cube/2.0/siamgcn |
| indoorcd | randla | indoorcd_cube2.0 | 2 | ablations/indoorcd/cube/2.0/randla |

## After the runs

Download each `ablations/<ds>/<sweep>/<value>/<model>/*.npz` back into the repo tree, then locally: `python src/build_ablation_tables.py --sweep both` and `python src/plot_ablations.py`. The builders pull the main anchors + ICP automatically and append the deep rows. predictions/ and main_table.csv are never touched.
