# FastPose-ViT on BOP: multi-object 6D pose

FastPose-ViT regresses the 6D pose of an object directly from an RGB crop around its 2D detection:
no 2D-3D correspondences, no PnP, no depth, no refinement. This document extends it from one
spacecraft (SPEED) to many objects on the [BOP benchmark](https://bop.felk.cvut.cz/): one shared
network for all objects of a dataset, trained on synthetic (PBR) images only, evaluated with the
official BOP19 metric (AR = mean of AR_VSD, AR_MSSD, AR_MSPD, `bop_toolkit`).

Status (2026-11-17): LM-O, YCB-V and T-LESS done, deployment (ONNX, TensorRT fp16, latency) done, our own
multi-class detector (M9 stretch, LM-O two seeds, YCB-V), the 224 px resolution ablation, the paper ablations of
`docs/PAPER_TODO.md` §3, M11 phase A and the 4× check of `tlw` + `occl` done (nothing adopted). Running (results not in
this document yet): an 8× budget on LM-O, a second seed of `tlw` + `occl` at 4×, and YCB-V trained on PBR + real images.

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
| *Ours, own LW-DETR small detector, full TensorRT fp16 pipeline* | | | *0.654* | *0.486* | *0.628* | *0.848* | *0.020* |
| *Ours, own LW-DETR small detector, PyTorch* | | | *0.654* | *0.486* | *0.627* | *0.848* | *0.035* |
| *Ours, own LW-DETR small detector, seed b, PyTorch* | | | *0.656* | *0.487* | *0.631* | *0.849* | |
| *Ours, own LW-DETR tiny detector (2 seeds), PyTorch* | | | *0.645 / 0.650* | *0.475 / 0.478* | *0.619 / 0.619* | *0.843 / 0.851* | *0.032* |

Own detector vs default: −0.014 AR (seed b: 0.656, −0.012), mostly MSPD (0.848 vs 0.873) and the smallest object (ape
0.696 → 0.645; 11.4 % of our ape boxes have IoU < 0.5 vs 1.7 % for the default detector); duck and
eggbox gain slightly. Per object: `$NAS_HOME/results/paper/per_object_ar_lmo_tok4x_ownsmall_trt.md`.

Our time is the BOP `time` of the evaluation run (A100, PyTorch, including the default detector's
81.6 ms, re-timed 2026-11-08 with the current per-image path; the M7 run gave 0.134 s with the
older path, same AR); see §3 for the deployment latency. Seeds: `tok4x` / `tok4x2` give
0.668 / 0.664 (mean 0.666). **Selection rule (audit W3, 2026-11-14):** the reported seed is the one
with the higher *validation* score ((AR_MSSD + AR_MSPD)/2, own recalls, GT boxes; scripts/val_score.py):
LM-O real LM val `tok4x` 0.8969 vs `tok4x2` 0.8909; YCB-V val_pbr `tok4x2` 0.8619 vs `tok4x` 0.8583.
Always quote the mean next to it (LM-O 0.666, YCB-V 0.698); the best-seed number is optimistic by
about 0.003.

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
| *Ours, own LW-DETR small detector (`tok4x2`), PyTorch (2026-11-16)* | | *0.681* | *0.574* | *0.663* | *0.805* | |

Reading (corrected 2026-11-11 after checking every submission page; the tables above are
incomplete, the verified ones are in `paper/sections/experiments.tex`): on **YCB-V** (PBR only) we
are 3rd, behind ZebraPose and GDRNPP, which train one network per object, and first among methods
with one model for all objects. On **LM-O** (PBR only) we are about 9th: PViT-6D (ViT direct
regression, one model, 0.740), RADet+PFA (0.745), MRPE (0.715), PoseDETR (single-stage direct, one
model, 0.695, 12 ms on an RTX 4090 with TensorRT), MRC-Net (0.685) and RACE-6D (0.683, 12 ms) are
above us. **The earlier claim "best direct-regression entry" is wrong.** PViT-6D (arXiv 2311.17504)
also already uses the size-invariant depth target we call `site` (its "SIZP", with s = max(w, h)),
a shared model for all objects with a class embedding, and allocentric 6D rotations. Our 2D alignment
(AR_MSPD) is at their level; the gap is in the 3D metrics (VSD, MSSD), dominated by the depth
error of the directly regressed translation.

### 1.3 RGB-D variant (M11 C1, 2026-11-14)

The RGB model's poses, with the distance corrected from the test depth image
(`scripts/depth_correction.py`, default `--method splat`: project the model points with the
predicted pose, keep the visible ones with a 4×4-pixel z-buffer, median of measured − predicted
depth over the points that agree within 20 / 10 / 5 % of t_z, shift t along its viewing ray; 3
iterations; rotation unchanged). GPU, all estimates of an image batched: **3–4 ms per image**
(the BOP-renderer reference, `--method render`, gives the same AR at ~85 ms). All runs:
`$NAS_HOME/results/m11/rgbd_splat/` (`scripts/m11_rgbd.sh`).

| RGB input | RGB AR | **RGB-D AR** | AR_VSD | AR_MSSD | AR_MSPD | Time (s) |
|---|---|---|---|---|---|---|
| LM-O `tok4x`, default det. | 0.668 | **0.749** | 0.592 | 0.786 | 0.870 | 0.139 |
| LM-O `tok4x2`, default det. | 0.664 | **0.752** | 0.596 | 0.786 | 0.874 | 0.104 |
| LM-O, own LW-DETR small + TensorRT | 0.654 | **0.738** | 0.590 | 0.777 | 0.847 | **0.024** |
| YCB-V `tok4x`, default det. | 0.694 | **0.849** | 0.795 | 0.904 | 0.847 | 0.124 |
| YCB-V `tok4x2`, default det. | 0.702 | **0.851** | 0.797 | 0.908 | 0.849 | 0.121 |
| YCB-V `tok4x2`, own LW-DETR small + PyTorch | 0.681 | **0.827** | 0.773 | 0.885 | 0.822 | |
| T-LESS `tok4x`, default det. | 0.762 | **0.858** | 0.789 | 0.853 | 0.931 | 0.101 |
| T-LESS `tok4x2`, default det. | 0.762 | **0.859** | 0.790 | 0.854 | 0.933 | 0.101 |

PBR-trained RGB-D leaderboard entries (training data checked on the LM-O submission pages
2026-11-14): LM-O HiPose 0.799 (per object, training data not stated), RADet+PFA 0.797 (one model,
1.7 s), GPose2023 0.794 (per object, 2.6 s), GDRNPP 0.775 (4.5 s), SurfEmb 0.760 (11.9 s),
RACE-6D 0.733 (one model, 0.013 s, **the same depth-shift idea**), RCVPose 0.729, FFB6D 0.687
(0.19 s); ZebraPose + ICP 0.752 left out (conflicting training data). YCB-V: HiPose 0.907, GDRNPP
0.906, GPose2023 0.901, RCVPose 0.843, RADet+PFA 0.826, SurfEmb 0.799, FFB6D 0.758. The depth shift
along the viewing ray is not new (RACE-6D RGB-D entry, GDRNPP "Fast" RGB-D entry): credit it.

### 1.4 T-LESS (test_primesense, BOP19 targets, 1,000 images, 6,423 instances of 30 objects)

One shared network for the 30 objects (`scripts/m10_tless.sh`: token, `domain_gap`, symmetric loss
with offsets, 384 px, 4× budget = 633,120 samples per epoch, 150 epochs, PBR images only, 2 seeds);
runs `tless-m10-tok4x-002` (seed a) and `tless-m10-tok4x2-002` (seed b), evaluated 2026-11-14 / 15
with `last.pt`.

| Input | AR (seed a / b) | AR_VSD | AR_MSSD | AR_MSPD | Time per image |
|---|---|---|---|---|---|
| RGB, default detections | **0.762 / 0.762** | 0.650 / 0.651 | 0.701 / 0.701 | 0.933 / 0.935 | 97 / 98 ms (with the detector) |
| RGB, GT boxes (oracle) | 0.813 / 0.812 | 0.706 / 0.705 | 0.754 / 0.753 | 0.978 / 0.978 | 16 ms |
| RGB-D, default detections (§1.3) | 0.858 / 0.859 | 0.789 / 0.790 | 0.853 / 0.854 | 0.931 / 0.933 | 101 ms |

- The detector costs 0.05 AR on T-LESS (LM-O 0.021, YCB-V 0.005). 9 of the 6,423 targets get no
  detection (6,414 estimates) and count as failures.
- In the per-object tables (`table_test.md`), the **translation means are not meaningful with
  detections**: a target without detection gets a pose 10 km away (`src/bop_evaluate.py`), which
  inflates the mean of objects 4, 14, 15 and 22 (up to 10^5 mm). Use the medians (11 mm on average).
- Depth correction (RGB-D): median shift −11 / −12 mm, 3.2 ms per image, 6,398 / 6,394 of 6,414
  estimates corrected.
- Leaderboard comparison: **not done yet** (PBR-only RGB entries of the T-LESS leaderboard and their
  training data have to be read from the submission pages).

Per object, seed b (`tok4x2`) with default detections (`scripts/bop_per_object_ar.py`, the
recomputed AR equals the official 0.7624; mean over objects 0.801):

| Object | AR | AR_VSD | AR_MSSD | AR_MSPD |
|---|---|---|---|---|
| 1 | 0.651 | 0.493 | 0.545 | 0.916 |
| 2 | 0.527 | 0.318 | 0.374 | 0.889 |
| 3 | 0.719 | 0.573 | 0.616 | 0.968 |
| 4 | 0.707 | 0.615 | 0.664 | 0.843 |
| 5 | 0.884 | 0.824 | 0.838 | 0.990 |
| 6 | 0.889 | 0.822 | 0.850 | 0.994 |
| 7 | 0.886 | 0.841 | 0.877 | 0.940 |
| 8 | 0.850 | 0.777 | 0.845 | 0.927 |
| 9 | 0.819 | 0.703 | 0.771 | 0.982 |
| 10 | 0.854 | 0.753 | 0.809 | 0.999 |
| 11 | 0.754 | 0.584 | 0.695 | 0.982 |
| 12 | 0.807 | 0.695 | 0.758 | 0.967 |
| 13 | 0.779 | 0.648 | 0.753 | 0.936 |
| 14 | 0.786 | 0.676 | 0.727 | 0.954 |
| 15 | 0.797 | 0.697 | 0.756 | 0.938 |
| 16 | 0.703 | 0.531 | 0.589 | 0.989 |
| 17 | 0.869 | 0.792 | 0.822 | 0.992 |
| 18 | 0.882 | 0.852 | 0.853 | 0.942 |
| 19 | 0.765 | 0.653 | 0.692 | 0.949 |
| 20 | 0.789 | 0.679 | 0.750 | 0.937 |
| 21 | 0.752 | 0.685 | 0.709 | 0.863 |
| 22 | 0.632 | 0.541 | 0.574 | 0.782 |
| 23 | 0.807 | 0.711 | 0.747 | 0.962 |
| 24 | 0.823 | 0.662 | 0.815 | 0.992 |
| 25 | 0.851 | 0.761 | 0.801 | 0.991 |
| 26 | 0.859 | 0.764 | 0.837 | 0.975 |
| 27 | 0.841 | 0.774 | 0.802 | 0.947 |
| 28 | 0.896 | 0.868 | 0.875 | 0.945 |
| 29 | 0.949 | 0.926 | 0.927 | 0.994 |
| 30 | 0.897 | 0.856 | 0.865 | 0.971 |
| **all instances** | 0.762 | 0.651 | 0.701 | 0.935 |
| mean over objects | 0.801 | | | |

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
| YCB-V conditioning (2×, detections) | per-object heads | 0.674 / 0.678 | 0.676 |
| | **per-object class token** | 0.681 / 0.689 | 0.685 |
| Symmetry-aware rotation loss (token, 2×) | off | 0.659 / 0.651 | 0.655 |
| | **on** | 0.655 / 0.655 | 0.655 |
| Training crop source (token, 2×, symmetric loss) | **`bbox_visib`** | 0.655 / 0.655 | 0.655 |
| | `bbox_obj` | 0.646 / 0.650 | 0.648 |
| | mix | 0.650 / 0.654 | 0.652 |
| Final model (token, 4×) | **`tok4x` / `tok4x2`** | **0.668 / 0.664** | **0.666** |
| One model per object (M10, same total compute) | 8 single-object models | 0.672 / 0.666 (GT boxes 0.688 / 0.682) | 0.669 (shared: 0.666) |
| Checkpoint (final model) | `model.pth` (best val) vs **`last.pt`** | 0.6686 vs 0.6685 (YCB-V 0.7021 vs 0.7021) | |
| Precision at deployment (M9) | PyTorch vs **TensorRT fp16** | 0.6685 vs 0.6685 (YCB-V 0.7021 vs 0.7017) | |
| Backbone resolution (M10) | **384 px** vs 224 px | 0.668 / 0.664 vs 0.643 / 0.643 | 0.666 vs 0.643 |
| **Paper ablations (M10)**: each = baseline token, 2×, symmetric loss, `domain_gap`, 10 % jitter, `site`, SWAG, apparent rotation (0.655 / 0.655) with one change | | | |
| Rotation target | **apparent (allocentric)** vs absolute (egocentric) | 0.546 / 0.548 | 0.547 (−0.108) |
| Pre-training | **SWAG** vs random init (scratch) | 0.516 / 0.517 | 0.517 (−0.138) |
| Depth target, 8 objects | **`site`** vs `legacy` (WACV α-based) | 0.610 / 0.625 | 0.617 (−0.038) |
| Bounding-box jitter (per side) | 0 % | 0.625 / 0.625 | 0.625 (−0.030) |
| | 5 % | 0.647 / 0.654 | 0.650 (−0.005) |
| | **10 %** | 0.655 / 0.655 | 0.655 |
| | 15 % | 0.655 / 0.651 | 0.653 (−0.002) |
| | 20 % | 0.650 / 0.651 | 0.650 (−0.005) |
| | 30 % | 0.646 / 0.643 | 0.644 (−0.011) |
| Pixel augmentation (study, 2026-11-14) | none | 0.626 / 0.629 | 0.627 |
| | color (brightness / contrast, gamma, hue-saturation) | 0.649 / 0.641 | 0.645 |
| | blur (Gaussian, motion, defocus) | 0.628 / 0.630 | 0.629 |
| | noise (Gaussian, ISO) | 0.627 / 0.627 | 0.627 |
| | color + blur + noise | 0.641 / 0.643 | 0.642 |
| | **`domain_gap`, p = 0.5** | **0.655 / 0.655** | **0.655** |
| | `domain_gap`, p = 0.8 | 0.641 / 0.631 | 0.636 |
| **M11 phase A (2026-11-15)**: baseline re-trained on the current code (`lmo-m12-base{,2}-001`; the 0.655 baseline ran older code) | | 0.650 / 0.654 | 0.652; validation score 0.873 |
| Context: image around the box instead of black bands (`--crop_context`) | no margin | 0.623 / 0.621 | 0.622 (−0.030), val 0.812 |
| | margin 1.25× | 0.624 / 0.627 | 0.625 (−0.027), val 0.816 |
| | margin 1.5× | 0.623 / 0.619 | 0.621 (−0.031), val 0.814 |
| Synthetic occlusion (`--bop_occlusion_aug`, 1–4 rectangles, p = 0.5) | | 0.657 / 0.651 | 0.654 (+0.002), val 0.876 (+0.003) |
| Translation loss weight ×2 (`--translation_loss_weight 2`) | | 0.659 / 0.658 | 0.659 (+0.007), val 0.874 (+0.001) |
| Occlusion + translation weight ×2 (`tlwocc`, A6, 2026-11-16) | | 0.664 / 0.658 | 0.661 (+0.009), val 0.882 (+0.009) |
| Same at 4× (A7, 2026-11-17, one seed) | 4× | 0.669 | +0.003 vs `tok4x` / `tok4x2` (0.666), val 0.887 (+0.004): not confirmed |

Augmentation reading: augmentation matters (none −0.028); only color helps alone; blur or noise
alone = none; `domain_gap` (adds gray, tone curve, CLAHE, solarize, sun flare, compression /
downscale) is best, but only at p = 0.5 (p = 0.8: −0.019). Box jitter: 5–20 % plateau (0.650–0.655),
0 % and 30 % worse. Kept: `domain_gap` p = 0.5, 10 %.

Reading (2026-11-08, replaces the earlier conclusions drawn from val ADD and driller ADD):
- **Kept, clear effect** (paper ablations, 2 seeds each, seed spread ≤ 0.002 except `legacy`
  0.016): SWAG pre-training (+0.14 AR vs scratch), the apparent (allocentric) rotation target
  (+0.11 AR vs absolute, the WACV contribution holds on BOP; every object loses, eggbox the most,
  0.32 → 0.18), `site` depth on all 8 objects (+0.04 AR vs `legacy`; the WACV α depth loses most
  on the driller 0.79 → 0.70 and the can 0.80 → 0.75, i.e. larger objects), bbox jitter (0 % costs
  0.030, 20 % ≈ 10 %, −0.005; WACV Table 8 found the same direction), the 384 px backbone (+0.023 AR vs 224 px, at
  about twice the pose time, §3), `site` depth (+0.06 AR on the driller vs `legacy`), training compute
  (1× → 2× → 4×: 0.639 → 0.650 → 0.661), class-token conditioning vs per-object heads (+0.005;
  heads + token "both" +0.010; both at about the seed spread; on YCB-V with 21 objects the token is +0.009 AR over heads, GT boxes 0.694 vs 0.680, validation 0.833 vs 0.811).
- **No measurable effect on test AR** (kept for other reasons): `domain_gap` augmentation (+0.002
  multi-object, −0.003 driller; it was chosen in M3 on driller ADD, +0.05, and on LM real val), the
  symmetric loss (needed for the YCB-V clamps, M8), the crop source (`bbox_visib` marginally best),
  the decoupled loss, one model per object, `model.pth` vs `last.pt`, fp16.
- **M11 phase A (2026-11-15), nothing adopted:** context around the box costs 0.03 AR on both
  test and validation (VSD and MSSD); occlusion and the doubled translation weight are within the
  seed spread of the re-trained baseline (0.004) on the validation score with perturbed boxes
  (`eval_valn`, the choice rule), so they are not kept; the translation weight is the only one
  above the spread on test (+0.007). **A6 (2026-11-16): the two together (`tlwocc`) give 0.664 /
  0.658 AR (+0.009) and validation 0.885 / 0.880 (+0.009), both above the spread and both seeds
  above both baseline seeds. **A7 (2026-11-17): at 4× the gain is gone (0.669 vs 0.668 / 0.664, validation 0.887 vs 0.886 / 0.880, +0.003 / +0.004, below the +0.005 rule): not adopted, the final models stay.**
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
| Own LW-DETR small, seed b, PyTorch | LM-O | 0.653 | 0.932 | 0.727 | | 17.9 ms (A100, bf16) |
| Own LW-DETR small, PyTorch | YCB-V | 0.702 | 0.859 | 0.815 | | 17.0 ms (A100, bf16) |
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
4.2 ms mean), the whole method runs at **55 FPS for 8 objects when the frame is already decoded**
(summed medians) for AR 0.654 (vs 0.668 with the default detector). **Measured end to end (audit W5,
2026-11-15, `optimization/bop_e2e_latency.py`, `$NAS_HOME/results/m9/e2e_clean/`, idle A100, all 200
LM-O test images, 7.2 objects on average, from the encoded PNG):** RGB 384 px **25.2 ms (40 FPS)**
(decode 7.8 + detector 3.4 + crops / network / recovery 14.8; 17.4 ms = 57 FPS without decoding), RGB 224 px
18.5 ms (54 FPS), RGB-D 384 px 30.0 ms (33 FPS), RGB-D 224 px 23.4 ms (43 FPS). The AR of this loop equals
the offline one (RGB 0.654, RGB-D 0.738). Jetson Orin numbers: not measured (no device).

## 4. Known limitations
- **Depth bias on real YCB-V images** (D8): every object is estimated ~2.4 % too far (median); the
  PBR renders do not match the real camera's apparent object size. Training on `train_real`
  would likely fix it; not done (PBR-only result by choice).
- T-LESS: the weakest objects are 2, 22 and 1 (AR 0.53, 0.63, 0.65, seed b); the detector costs 0.05 AR
  (the highest of the three datasets).
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
