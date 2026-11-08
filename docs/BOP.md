# FastPose-ViT on BOP: multi-object 6D pose

FastPose-ViT regresses the 6D pose of an object directly from an RGB crop around its 2D detection:
no 2D-3D correspondences, no PnP, no depth, no refinement. This document extends it from one
spacecraft (SPEED) to many objects on the [BOP benchmark](https://bop.felk.cvut.cz/): one shared
network for all objects of a dataset, trained on synthetic (PBR) images only, evaluated with the
official BOP19 metric (AR = mean of AR_VSD, AR_MSSD, AR_MSPD, `bop_toolkit`).

Status (2026-11-08): LM-O and YCB-V done, deployment (ONNX, TensorRT fp16, latency) done. Still
running: our own multi-class detector (M9 stretch) and the 224 px resolution ablation; their rows
are marked *pending*.

## 1. Results

### 1.1 LM-O (test, BOP19 targets, 200 images, 1,445 instances)

RGB-only entries trained on PBR images, from the
[BOP leaderboard](https://bop.felk.cvut.cz/leaderboards/pose-estimation-bop19/lm-o/) (read
2026-10-28 with an automatic page extractor: spot-check a value on the page before quoting it).
"Refinement" = an iterative render-and-compare stage after the first estimate; "?" = not checked.
Times are per image as submitted, on each author's hardware.

| Method | Pose from | Refinement | AR | AR_VSD | AR_MSSD | AR_MSPD | Time (s) |
|---|---|---|---|---|---|---|---|
| ZebraPoseSAT-EffnetB4 (PBR, default det. 2023) | correspondences + PnP | no | 0.729 | 0.562 | 0.736 | 0.888 | 0.250 |
| GDRNPP-PBR-RGB-MModel | geometry-guided regression, one model per object | no | 0.713 | 0.549 | 0.701 | 0.887 | 0.206 |
| **FastPose-ViT (ours), default detections** | **direct regression, one model** | **no** | **0.668** | 0.491 | 0.641 | 0.873 | 0.134 |
| SurfEmb-PBR-RGB | dense correspondences + PnP | yes | 0.663 | 0.497 | 0.640 | 0.851 | 11.8 |
| CosyPose-ECCV20-PBR-1VIEW | render-and-compare | yes | 0.633 | 0.480 | 0.606 | 0.812 | 0.550 |
| CDPNv2_BOP20 (PBR, RGB) | correspondences + PnP | no | 0.624 | 0.445 | 0.612 | 0.815 | 0.163 |
| EPOS-BOP20-PBR | correspondences + PnP | no | 0.547 | 0.389 | 0.501 | 0.750 | 0.468 |
| *Ours, GT boxes (oracle, upper bound of the detector)* | | | *0.689* | *0.514* | *0.660* | *0.892* | *0.053* |
| *Ours, TensorRT fp16 engine, default detections* | | | *0.6685* | *0.4911* | *0.6415* | *0.8729* | |
| *Ours, own LW-DETR detector* | | | *pending* | | | | |

Our time is the BOP `time` of the evaluation run (A100, PyTorch, including the default detector's
81.6 ms); see §3 for the deployment latency.

### 1.2 YCB-V (test, BOP19 targets, 900 images, 4,123 instances)

RGB-only entries trained on PBR images only, from the
[BOP leaderboard](https://bop.felk.cvut.cz/leaderboards/pose-estimation-bop19/ycb-v/) (read
2026-11-08, same extractor and caveat). Most YCB-V entries also train on the real `train_real`
images (best RGB: 0.86); we do not (open issue D8, see §4).

| Method | Refinement | AR | AR_VSD | AR_MSSD | AR_MSPD | Time (s) |
|---|---|---|---|---|---|---|
| ZebraPoseSAT-EffnetB4 (PBR only, default det. 2023) | no | 0.729 | 0.640 | 0.725 | 0.821 | 0.250 |
| GDRNPP-PBR-RGB-MModel | no | 0.713 | 0.599 | 0.708 | 0.831 | 0.277 |
| **FastPose-ViT (ours), default detections** | **no** | **0.702** | 0.591 | 0.681 | 0.834 | 0.118 |
| ZebraPoseSAT-EffnetB4 (PBR only) | no | 0.691 | 0.607 | 0.686 | 0.780 | – |
| MRC-Net (PBR, default det. 2022) | ? | 0.681 | 0.572 | 0.666 | 0.804 | 1.134 |
| PoseDETR (PBR-RGB) | ? | 0.679 | 0.578 | 0.682 | 0.779 | 0.012 |
| SurfEmb-PBR-RGB | yes | 0.647 | 0.548 | 0.620 | 0.773 | 5.427 |
| SC6D (PBR only) | ? | 0.612 | 0.506 | 0.589 | 0.741 | 0.370 |
| CosyPose-ECCV20-PBR-1VIEW | yes | 0.574 | 0.516 | 0.554 | 0.653 | 0.342 |
| EPOS-BOP20-PBR | no | 0.499 | 0.411 | 0.464 | 0.621 | 0.764 |
| *Ours, GT boxes (oracle)* | | *0.707* | | | | |
| *Ours, TensorRT fp16 engine, default detections* | | *0.7017* | *0.5908* | *0.6806* | *0.8338* | |

Reading: on both datasets FastPose-ViT is the best direct-regression entry and within 0.01–0.05 AR
of the best PBR-only RGB methods, with one network for all objects and no PnP. Our 2D alignment
(AR_MSPD) is at their level; the gap is in the 3D metrics (VSD, MSSD), dominated by the depth
error of the directly regressed translation.

## 2. Ablations

All on LM-O unless stated; each row is one change from the final recipe. Two seeds per config
where available ("a / b"): the seed spread is ~0.005–0.015, so smaller differences are noise.
"val ADD" = mean ADD(-S)@0.1d recall over the 8 objects on LM real val with GT boxes (200
instances per object); "test AR" = official BOP19 AR on LM-O test with the default detections.

| Choice (milestone) | Options | Result | Kept |
|---|---|---|---|
| Depth parametrisation (M3, driller only, LM-O test ADD) | `site` / legacy | 0.760 / 0.770 vs 0.715 | `site` |
| Translation loss (M3, driller, LM-O test ADD) | relative / decoupled | 0.760 / 0.770 vs 0.760 | relative |
| Pixel augmentation (M3 driller test ADD; M4 8 objects val ADD) | `standard` / `domain_gap` | 0.765 vs **0.815**; 0.559 / 0.574 vs **0.631 / 0.630** | `domain_gap` |
| Training compute (M4 val ADD; M8 YCB-V test AR) | 1× / 2× / 4× samples per epoch | 0.631 / 0.630, 0.655 / 0.664, 0.663; YCB-V 2× 0.681 / 0.689 vs 4× **0.694 / 0.702** | 4× |
| Object conditioning (M4, 2×, val ADD) | per-object heads / per-object class token / both | 0.655 / 0.664, **0.679 / 0.670**, 0.660 / 0.661 | class token |
| Symmetry-aware rotation loss (M5, token 2×, val ADD) | off / on | 0.679 / 0.670 vs 0.658 / 0.670 (no gain on eggbox, glue) | on for BOP (user), required for YCB-V clamps |
| Training crop source (M7, token 2×, test AR) | `bbox_visib` / `bbox_obj` / mix | **0.654 / 0.655**, 0.646 / 0.650, 0.650 / 0.654 | `bbox_visib` |
| One model per object (M10, test AR) | shared / 8 single-object models (same total compute) | 0.668 vs 0.672 (GT boxes 0.689 vs 0.688) | shared |
| Backbone resolution (M10, test AR) | 384 px / 224 px | 0.668 / – vs *pending* | 384 px |
| Precision at deployment (M9, test AR) | PyTorch / TensorRT fp16 | 0.6685 vs 0.6685 (YCB-V 0.7021 vs 0.7017) | TensorRT fp16 |
| Detector (M9 stretch, test AR) | BOP default (GDRNPPDet) / own LW-DETR tiny, small | 0.668 vs *pending* | |

Final recipe: ViT-B/16 at 384 px (SWAG ImageNet weights), one merged linear head (3 translation +
6D rotation values), per-object class token, `site` depth with relative translation loss,
6D relative-Frobenius rotation loss (symmetry-aware for BOP), Muon optimizer, cosine warm
restarts, 150 epochs, batch 32, `domain_gap` augmentation, crops around `bbox_visib` with 10 %
margin, object-balanced epochs (LM-O: 168,832 samples per epoch = 4×).

## 3. Deployment and latency (M9)

The network is exported to ONNX (dynamic batch: one crop per object of an image, plus the object
index of each crop) and built as a TensorRT engine (fp16, one profile for 1–16 crops). Outputs:
ONNX Runtime vs PyTorch fp32 max 3e-6, TensorRT fp32 7.6e-6, fp16 2e-2 (in the 6D rotation values),
which costs ≤ 0.0004 AR (§1).

Latency per image on an NVIDIA A100 (16 CPU cores), LM-O model, median of 200 runs (p90 within
0.3 ms). Network = the N crops already on the GPU; pipeline = one decoded image to N poses on the
CPU (crop and normalise every instance, network, R, t recovery).

| ms per image | N = 1 | N = 4 | N = 8 | N = 16 |
|---|---|---|---|---|
| Network, PyTorch fp16 autocast | 7.2 | 7.6 | 8.7 | 15.9 |
| Network, **TensorRT fp16** | **1.8** | **3.5** | **5.7** | **11.0** |
| Pipeline, PyTorch fp16 autocast | 10.8 | 13.5 | 17.0 | 29.5 |
| Pipeline, **TensorRT fp16** | **5.3** | **9.3** | **14.2** | **24.0** |
| FPS, TensorRT pipeline (pose stage) | 189 | 108 | 70 | 42 |
| *224 px backbone: network, TensorRT fp16* | *1.4* | *1.7* | *2.2* | *3.9* |
| *224 px backbone: pipeline, TensorRT fp16* | *4.4* | *6.1* | *7.6* | *11.1* |
| FPS including the BOP default detector (81.6 ms per image) | 11.5 | 11.0 | 10.4 | 9.5 |

The pose stage is real time (70 FPS for 8 objects; 130 FPS with the 224 px backbone, whose AR is
*pending*, §2). With the default detector (GDRNPPDet, time
reported by its authors), the whole method runs at ~10 FPS: the detector is the bottleneck, hence
our own LW-DETR detector (*pending*). Jetson Orin numbers: not measured (no device).

## 4. Known limitations
- **Depth bias on real YCB-V images** (D8): every object is estimated ~2.4 % too far (median); the
  PBR renders do not match the real camera's apparent object size. Training on `train_real`
  would likely fix it; not done (PBR-only result by choice).
- Heavily occluded objects (LM-O eggbox, duck) are the weakest: direct regression from a crop has
  no explicit visibility reasoning.
- All numbers use one GPU type (A100); Jetson latency is not measured.

## 5. Reproduce

Data: BOP datasets in the standard layout (`scripts/make_bop_tree.sh` builds it from the cluster
copy), BOP default detections under `$NAS_HOME/datasets/bop_detections/`, `bop_toolkit` cloned
next to the repo. All commands from the repo root with `PYTHONPATH=.`.

| Step | Command |
|---|---|
| Train LM-O final model | `bash scripts/m4_multi.sh tok4x` (inside a training job: `$RUN_DIR`) |
| Train YCB-V final model | `bash scripts/m8_ycbv.sh tok4x2` |
| Predict + own metrics, default detections | `python src/bop_evaluate.py --dataset lmo --model_weights RUN/tok4x/last.pt --object_conditioning token --splits test --detections DET.json --out_dir OUT` |
| Official BOP19 AR | `bash scripts/bop_eval.sh OUT fastposevitmo_lmo-test.csv` |
| ONNX export + check | `python optimization/bop_export.py --dataset lmo --model_weights RUN/tok4x/last.pt --onnx_path OUT/model.onnx` |
| TensorRT engine | `python optimization/bop_trt.py --onnx_path OUT/model.onnx --engine_path OUT/model_fp16.engine --fp16` |
| AR through the engine | `bash scripts/m9_trt_eval.sh OUT/model_fp16.engine` (`DATASET=ycbv` for YCB-V) |
| Latency table | `python optimization/bop_latency.py --model_weights RUN/tok4x/last.pt --engine OUT/model_fp16.engine --out_dir OUT/latency` |
| Own detector | `bash scripts/m9_det.sh tiny 0` (train, LM-O test detections, pose eval) |

The full log of every experiment (configs, run folders, per-object tables) is in `TASKS.md`.
