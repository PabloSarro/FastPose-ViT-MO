# FastPose-ViT on BOP: multi-object 6D pose

FastPose-ViT regresses the 6D pose of an object directly from an RGB crop around its 2D detection:
no 2D-3D correspondences, no PnP, no depth, no refinement. This document extends it from one
spacecraft (SPEED) to many objects on the [BOP benchmark](https://bop.felk.cvut.cz/): one shared
network for all objects of a dataset, trained on synthetic (PBR) images only, evaluated with the
official BOP19 metric (AR = mean of AR_VSD, AR_MSSD, AR_MSPD, `bop_toolkit`).

Status (2026-11-09): LM-O and YCB-V done, deployment (ONNX, TensorRT fp16, latency) done, our own
multi-class detector (M9 stretch) and the 224 px resolution ablation done. Still running: the
paper ablations of `docs/PAPER_TODO.md` §3 and T-LESS; their rows are marked *pending*.

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
| **FastPose-ViT (ours), default detections** | **direct regression, one model** | **no** | **0.668** | 0.491 | 0.641 | 0.873 | 0.099 |
| SurfEmb-PBR-RGB | dense correspondences + PnP | yes | 0.663 | 0.497 | 0.640 | 0.851 | 11.8 |
| CosyPose-ECCV20-PBR-1VIEW | render-and-compare | yes | 0.633 | 0.480 | 0.606 | 0.812 | 0.550 |
| CDPNv2_BOP20 (PBR, RGB) | correspondences + PnP | no | 0.624 | 0.445 | 0.612 | 0.815 | 0.163 |
| EPOS-BOP20-PBR | correspondences + PnP | no | 0.547 | 0.389 | 0.501 | 0.750 | 0.468 |
| *Ours, GT boxes (oracle, upper bound of the detector)* | | | *0.689* | *0.514* | *0.660* | *0.892* | *0.053* |
| *Ours, TensorRT fp16 engine, default detections* | | | *0.6685* | *0.4911* | *0.6415* | *0.8729* | |
| *Ours, own LW-DETR small detector, full TensorRT fp16 pipeline* | | | *0.654* | | | | *see §3* |
| *Ours, own LW-DETR small / tiny detector, PyTorch* | | | *0.654 / 0.645–0.650* | | | | |

Our time is the BOP `time` of the evaluation run (A100, PyTorch, including the default detector's
81.6 ms, re-timed 2026-11-08 with the current per-image path; the M7 run gave 0.134 s with the
older path, same AR); see §3 for the deployment latency. Seeds: `tok4x` / `tok4x2` give
0.669 / 0.664 (mean 0.666).

### 1.2 YCB-V (test, BOP19 targets, 900 images, 4,123 instances)

RGB-only entries trained on PBR images only, from the
[BOP leaderboard](https://bop.felk.cvut.cz/leaderboards/pose-estimation-bop19/ycb-v/) (read
2026-11-08, same extractor and caveat). Most YCB-V entries also train on the real `train_real`
images (best RGB: 0.86); we do not (open issue D8, see §4).

| Method | Refinement | AR | AR_VSD | AR_MSSD | AR_MSPD | Time (s) |
|---|---|---|---|---|---|---|
| ZebraPoseSAT-EffnetB4 (PBR only, default det. 2023) | no | 0.729 | 0.640 | 0.725 | 0.821 | 0.250 |
| GDRNPP-PBR-RGB-MModel | no | 0.713 | 0.599 | 0.708 | 0.831 | 0.277 |
| **FastPose-ViT (ours), default detections** | **no** | **0.702** | 0.591 | 0.681 | 0.834 | 0.094 |
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

All ablations on **one metric: the official BOP19 AR on LM-O test with the BOP default
detections**, each model's `last.pt` (re-evaluated 2026-11-08 with `scripts/paper_ablation_eval.sh`;
raw table `$NAS_HOME/results/paper/ablation_ar_raw.md`). Two seeds per config ("a / b"); the seed
spread is up to 0.016 AR (`std` / `std2`), typically 0.005–0.008, so smaller differences are noise.
Each block changes one thing; the bold row is the choice kept.

**Single-object (driller, M3, 1× compute, `standard` aug unless stated)**: AR over the driller only
(driller-only CSV and targets file), comparable within this block only.

| Change | AR | AR_VSD | AR_MSSD | AR_MSPD |
|---|---|---|---|---|
| **`site` depth, relative translation loss (B1, 2 seeds)** | **0.844 / 0.845** | 0.726 / 0.719 | 0.873 / 0.884 | 0.934 / 0.932 |
| `legacy` depth (the WACV α-based U_z) | 0.785 | 0.608 | 0.852 | 0.896 |
| decoupled translation loss | 0.844 | 0.719 | 0.881 | 0.934 |
| `domain_gap` augmentation | 0.841 | 0.712 | 0.884 | 0.928 |

**Shared 8-object model (M4–M7)**

| Block | Config | AR (seed a / b) | Mean |
|---|---|---|---|
| Augmentation (heads, 1×) | `standard` | 0.629 / 0.645 | 0.637 |
| | **`domain_gap`** | 0.641 / 0.637 | 0.639 |
| Training compute (heads, `domain_gap`) | 1× | 0.641 / 0.637 | 0.639 |
| | 2× | 0.650 / 0.650 | 0.650 |
| | **4×** | 0.661 (1 seed) | 0.661 |
| Object conditioning (`domain_gap`, 2×) | per-object heads | 0.650 / 0.650 | 0.650 |
| | **per-object class token** | 0.659 / 0.651 | 0.655 |
| | both | 0.660 / 0.661 | 0.660 |
| Symmetry-aware rotation loss (token, 2×) | off | 0.659 / 0.651 | 0.655 |
| | **on** | 0.655 / 0.655 | 0.655 |
| Training crop source (token, 2×, symmetric loss) | **`bbox_visib`** | 0.655 / 0.655 | 0.655 |
| | `bbox_obj` | 0.646 / 0.650 | 0.648 |
| | mix | 0.650 / 0.654 | 0.652 |
| Final model (token, 4×) | **`tok4x` / `tok4x2`** | **0.669 / 0.664** | **0.666** |
| One model per object (M10, same total compute) | 8 single-object models | 0.672 (1 seed) | |
| Checkpoint (final model) | `model.pth` (best val) vs **`last.pt`** | 0.6686 vs 0.6685 (YCB-V 0.7021 vs 0.7021) | |
| Precision at deployment (M9) | PyTorch vs **TensorRT fp16** | 0.6685 vs 0.6685 (YCB-V 0.7021 vs 0.7017) | |
| Backbone resolution (M10) | **384 px** vs 224 px | 0.669 / 0.664 vs 0.643 / 0.643 | 0.666 vs 0.643 |
| **Paper ablations (M10)**: each = baseline token, 2×, symmetric loss, `domain_gap`, 10 % jitter, `site`, SWAG, apparent rotation (0.655 / 0.655) with one change | | | |
| Rotation target | **apparent (allocentric)** vs absolute (egocentric) | 0.546 / 0.548 | 0.547 (−0.108) |
| Pre-training | **SWAG** vs random init (scratch) | 0.516 / 0.517 | 0.517 (−0.138) |
| Depth target, 8 objects | **`site`** vs `legacy` (WACV α-based) | 0.610 / 0.625 | 0.617 (−0.038) |
| Bounding-box jitter (per side) | 0 % | 0.625 / 0.625 | 0.625 (−0.030) |
| | **10 %** | 0.655 / 0.655 | 0.655 |
| | 20 % | 0.650 / 0.651 | 0.650 (−0.005) |
| | 5 / 15 / 30 % | *pending* | |

Reading (2026-11-08, replaces the earlier conclusions drawn from val ADD and driller ADD):
- **Kept, clear effect** (paper ablations, 2 seeds each, seed spread ≤ 0.002 except `legacy`
  0.016): SWAG pre-training (+0.14 AR vs scratch), the apparent (allocentric) rotation target
  (+0.11 AR vs absolute, the WACV contribution holds on BOP; every object loses, eggbox the most,
  0.32 → 0.18), `site` depth on all 8 objects (+0.04 AR vs `legacy`; the WACV α depth loses most
  on the driller 0.79 → 0.70 and the can 0.80 → 0.75, i.e. larger objects), bbox jitter (0 % costs
  0.030, 20 % ≈ 10 %, −0.005; WACV Table 8 found the same direction), the 384 px backbone (+0.023 AR vs 224 px, at
  about twice the pose time, §3), `site` depth (+0.06 AR on the driller vs `legacy`), training compute
  (1× → 2× → 4×: 0.639 → 0.650 → 0.661), class-token conditioning vs per-object heads (+0.005;
  heads + token "both" +0.010; both at about the seed spread).
- **No measurable effect on test AR** (kept for other reasons): `domain_gap` augmentation (+0.002
  multi-object, −0.003 driller; it was chosen in M3 on driller ADD, +0.05, and on LM real val), the
  symmetric loss (needed for the YCB-V clamps, M8), the crop source (`bbox_visib` marginally best),
  the decoupled loss, one model per object, `model.pth` vs `last.pt`, fp16.
- The earlier metrics overstated the augmentation and conditioning effects; the paper should use
  this table.

### 2.1 Per object (official BOP19 AR, final models)

Computed from the BOP toolkit's per-object recalls (`scripts/bop_per_object_ar.py`; the
recomputed overall AR equals the official one). "all instances" is the official AR; the mean over
objects weighs every object equally.

LM-O `tok4x`, default detections:

| Object | AR | AR_VSD | AR_MSSD | AR_MSPD |
|---|---|---|---|---|
| 1 ape | 0.696 | 0.486 | 0.684 | 0.917 |
| 5 can | 0.826 | 0.685 | 0.850 | 0.944 |
| 6 cat | 0.632 | 0.434 | 0.568 | 0.895 |
| 8 driller | 0.814 | 0.675 | 0.862 | 0.906 |
| 9 duck | 0.644 | 0.499 | 0.541 | 0.892 |
| 10 eggbox | 0.357 | 0.169 | 0.263 | 0.639 |
| 11 glue | 0.648 | 0.459 | 0.624 | 0.861 |
| 12 holepuncher | 0.690 | 0.475 | 0.682 | 0.912 |
| **all instances** | 0.668 | 0.491 | 0.641 | 0.873 |
| mean over objects | 0.663 | | | |

LM-O `tok4x`, GT boxes (oracle):

| Object | AR | AR_VSD | AR_MSSD | AR_MSPD |
|---|---|---|---|---|
| 1 ape | 0.715 | 0.511 | 0.706 | 0.929 |
| 5 can | 0.836 | 0.705 | 0.862 | 0.940 |
| 6 cat | 0.644 | 0.448 | 0.571 | 0.912 |
| 8 driller | 0.827 | 0.696 | 0.871 | 0.914 |
| 9 duck | 0.670 | 0.532 | 0.579 | 0.900 |
| 10 eggbox | 0.432 | 0.230 | 0.321 | 0.744 |
| 11 glue | 0.669 | 0.480 | 0.654 | 0.874 |
| 12 holepuncher | 0.681 | 0.466 | 0.662 | 0.913 |
| **all instances** | 0.689 | 0.514 | 0.660 | 0.892 |
| mean over objects | 0.684 | | | |

YCB-V `tok4x2`, default detections:

| Object | AR | AR_VSD | AR_MSSD | AR_MSPD |
|---|---|---|---|---|
| 1 master_chef_can | 0.652 | 0.553 | 0.592 | 0.810 |
| 2 cracker_box | 0.811 | 0.768 | 0.822 | 0.843 |
| 3 sugar_box | 0.767 | 0.672 | 0.759 | 0.870 |
| 4 tomato_soup_can | 0.662 | 0.533 | 0.608 | 0.845 |
| 5 mustard_bottle | 0.771 | 0.666 | 0.766 | 0.881 |
| 6 tuna_fish_can | 0.741 | 0.617 | 0.691 | 0.914 |
| 7 pudding_box | 0.811 | 0.654 | 0.831 | 0.947 |
| 8 gelatin_box | 0.716 | 0.593 | 0.620 | 0.935 |
| 9 potted_meat_can | 0.520 | 0.444 | 0.462 | 0.655 |
| 10 banana | 0.835 | 0.757 | 0.890 | 0.859 |
| 11 pitcher_base | 0.797 | 0.789 | 0.836 | 0.764 |
| 12 bleach_cleanser | 0.776 | 0.731 | 0.796 | 0.803 |
| 13 bowl | 0.741 | 0.646 | 0.791 | 0.787 |
| 14 mug | 0.648 | 0.487 | 0.575 | 0.883 |
| 15 power_drill | 0.772 | 0.675 | 0.789 | 0.852 |
| 16 wood_block | 0.386 | 0.232 | 0.233 | 0.692 |
| 17 scissors | 0.411 | 0.153 | 0.261 | 0.817 |
| 18 large_marker | 0.571 | 0.322 | 0.479 | 0.911 |
| 19 large_clamp | 0.643 | 0.438 | 0.657 | 0.835 |
| 20 extra_large_clamp | 0.689 | 0.532 | 0.771 | 0.764 |
| 21 foam_brick | 0.536 | 0.316 | 0.376 | 0.917 |
| **all instances** | 0.702 | 0.591 | 0.681 | 0.834 |
| mean over objects | 0.679 | | | |

YCB-V `tok4x2`, GT boxes: `$NAS_HOME/results/paper/per_object_ar_ycbv_tok4x2_gt.md` (AR 0.707).

### 2.2 Model size and compute

Per crop, counted with `torch.utils.flop_counter` plus the attention products that it skips inside
SDPA (`$NAS_HOME/results/paper/model_size.py`). "GMACs" is the convention of the WACV paper's
Table 1 (which calls them GFLOPS: 17.5 / 55.5); FLOPs = 2 × MACs. TensorRT fp16 engine: 167 MB
(ONNX fp32: 330 MB).

| Input | Objects | Conditioning | Params (M) | GMACs per crop (WACV 'GFLOPS') | GFLOPs per crop |
|---|---|---|---|---|---|
| 224 px | 8 | token | 85.81 | 17.6 | 35.1 |
| 224 px | 21 | token | 85.82 | 17.6 | 35.1 |
| 224 px | 8 | heads | 85.85 | 17.6 | 35.1 |
| 384 px | 8 | token | 86.10 | 55.5 | 111.0 |
| 384 px | 21 | token | 86.11 | 55.5 | 111.0 |
| 384 px | 8 | heads | 86.15 | 55.5 | 111.0 |

One shared model adds ≤ 0.05 M parameters over the single-object network (class tokens and the
wider head), whatever the number of objects; 8 single-object models would need 8 × 86 M.

### 2.3 Detection quality (2D)

COCO bbox mAP against the test targets (`bbox_visib`, `scripts/detection_map.py`) and, per target,
the box the pose evaluation uses (top-1 detection of that object, `scripts/detection_iou.py`).

| Detections | Dataset | mAP | AP50 | AR100 | Target box IoU < 0.5 | Time per image |
|---|---|---|---|---|---|---|
| BOP default (GDRNPPDet-PBR) | LM-O | 0.657 | 0.929 | 0.721 | 6.3 % (eggbox 27 %, cat 13 %) | 81.6 ms (authors) |
| BOP default (GDRNPPDet-PBR) | YCB-V | 0.758 | 0.962 | 0.813 | ≤ 6.7 % per object | 78.8 ms (authors) |
| Own LW-DETR tiny (2 seeds), PyTorch | LM-O | 0.628 / 0.631 | 0.925 / 0.930 | 0.703 / 0.706 | | 14.3 / 14.6 ms (A100, bf16) |
| Own LW-DETR small, PyTorch | LM-O | 0.656 | 0.936 | 0.727 | | 16.8 ms (A100, bf16) |
| Own LW-DETR small, **TensorRT fp16** | LM-O | 0.657 | | 0.728 | 7.6 % (eggbox 21 %, cat 14 %) | **3.3 ms** (A100, incl. resize, top-k) |

Reading: the detector explains part of the GT-box → detection gap (LM-O AR 0.689 → 0.668): the
eggbox, our weakest object, also has the worst detections (27 % wrong boxes).

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
| *224 px backbone: pipeline, TensorRT fp16* | *4.3* | *5.8* | *7.1* | *10.6* |
| FPS including the BOP default detector (81.6 ms per image) | 11.5 | 11.0 | 10.4 | 9.5 |
| **FPS end to end with our LW-DETR small, TensorRT fp16 (4.2 ms mean per image), 384 px** | **105** | **74** | **55** | **35** |
| FPS end to end with our LW-DETR small, TensorRT fp16, 224 px | 118 | 100 | 88 | 68 |

End-to-end rows (re-measured 2026-11-11, `$NAS_HOME/results/m9/det-small-trt/latency_{384,224}/`)
= the pose pipeline median + the detector engine's mean time per image over LM-O test (incl.
resize, normalisation, top-k; median 3.3 ms), same GPU, run one after the other. Our
detector's engine needs the deformable-attention sampling written with gathers: TensorRT 10.14's
GridSample is wrong for it (fp32 mAP 0.633 instead of 0.658); with gathers, engine mAP = PyTorch.

The pose stage is real time (70 FPS for 8 objects; 130 FPS with the 224 px backbone, −0.023 AR,
§2). With the default detector (GDRNPPDet, time reported by its authors), the whole method runs
at ~10 FPS: the detector is the bottleneck. With our own LW-DETR small detector (TensorRT fp16,
4.2 ms mean), the whole method runs at **55 FPS for 8 objects** for AR 0.654 (vs 0.668 with the default
detector). Jetson Orin numbers: not measured (no device).

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
| Own detector | `bash scripts/m9_det.sh small 0` (train, LM-O test detections, pose eval) |
| Detector TensorRT engine + detections | `python optimization/bop_det_trt.py --variant small --weights RUN/small/best.pth --out_dir OUT` |

The full log of every experiment (configs, run folders, per-object tables) is in `TASKS.md`.
