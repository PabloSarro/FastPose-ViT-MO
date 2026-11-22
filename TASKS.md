# TASKS — Multi-object FastPose-ViT on BOP

Tracker for extending FastPose-ViT (single-object spacecraft pose) to **multi-object 6D pose estimation on BOP**.
Keep it current: update §1 (status), §9 (results) and §10 (next steps) at the end of every work session.

- Created: 2026-10-20 · Owner: Pablo Sarro · Branch for code: `multi-object` (created 2026-10-20 from `main` @ `6b3c430`, pushed to `origin` = `PabloSarro/FastPose-ViT-MO`)
- Paths: `$NAS_HOME` = `/mnt/cvlab/scratch/cvlab/home/sarro`, `$DATA_DIR` = `/mnt/cvlab/scratch/cvlab/home/pvancey/datasets`
- Status legend: `[ ]` todo · `[~]` in progress · `[x]` done · `[!]` blocked

---

## 0. Goal and scope

**Goal:** turn FastPose-ViT, a ViT that directly regresses the 6D pose of one object from a bbox crop of one RGB image, into a model that estimates the poses of **several objects per RGB image**, and measure it with the **official BOP metrics**.

| Decision | Value (decided 2026-10-20) |
|---|---|
| "Multi-image" means | Multi-object: several objects/instances per RGB image (not multi-view or temporal) |
| Success metric | BOP19 AR = mean(AR_VSD, AR_MSSD, AR_MSPD) via `bop_toolkit` on `test_targets_bop19.json` |
| ~~Must keep~~ | ~~RGB only (no depth) · direct regression (no PnP, no render-and-compare refinement) · real-time / Jetson deployable~~ · SPEED keeps working (score 0.0093). **Replaced 2026-11-18 by the rows below (supervisor meeting); SPEED must still keep working.** |
| **Goal (2026-11-18)** | **Best BOP19 AR on LM-O, YCB-V and T-LESS. Inference cost and model size do not matter** (supervisor: accuracy is what counts, even if the model gets much more expensive; a "small" or "cheap and fast" positioning is not attractive enough) |
| **Allowed (2026-11-18)** | PnP, render-and-compare refinement, one network per object, larger backbones and input resolution, ensembles. RGB stays the main input; the RGB-D block (depth correction) is kept as a separate block |
| **Training data (2026-11-18)** | Both **PBR only** and **PBR + real**, always labelled. Compare with the general leaderboard (both categories) and report the category where we place best (e.g. top 3 among PBR-only and/or PBR + real) |
| **Adoption rule (2026-11-18)** | Adopt a change when its **2-seed mean beats the baseline's 2-seed mean on both the noisy-box validation and the test AR, by any margin** (replaces "+0.005 above the seed spread"). Validation: LM-O `eval_valn`; YCB-V `eval_valrealn` (held-out real videos); T-LESS a real validation from `train_primesense` (M17) |
| Speed (2026-11-18) | Jetson dropped for the semester project (a limitation sentence). Times of other methods only as context, no speed ranking claim |
| Datasets | **LM-O first** (train on LM `train_pbr`), then YCB-V, then T-LESS |
| Timeline | Semester project. **EPFL report due 2027-01-01**: no page limit (usually ~20 pages), authors **Pablo Sarró Sánchez, Mathieu Salzmann** (not Pierre), written on the `paper/` draft. A later publication: Pablo Sarró Sánchez, Mathieu Salzmann, **Pierre Ancey**, with the conference's page limit |
| Release (2026-11-18) | Not required. If done: code on GitHub, weights on Hugging Face |

Week calendar (Mondays): W1 Oct 19 · W2 Oct 26 · W3 Nov 2 · W4 Nov 9 · W5 Nov 16 · W6 Nov 23 · W7 Nov 30 · W8 Dec 7 ·
W9 Dec 14 · W10 Dec 21 · W11 Dec 28 · W12 Jan 4 · W13 Jan 11 · W14 Jan 18. W10–W11 are holidays, a good slot for long runs.
**Schedule to the report (2026-11-18):** W5–W7 experiments M15–M19 · **decision gate Fri 2026-12-04** (final recipe per dataset) · W8–W9 final models (3 datasets × PBR only / PBR + real × 2 seeds, RGB-D) · W9–W11 report · **due 2027-01-01**.

---

## 1. Status board

| Milestone | Weeks | Status | Exit criterion | Result |
|---|---|---|---|---|
| M0 Setup and data access | W1 | `[x]` | Golden SPEED eval reproduced (0.0093) and bop_toolkit gives AR ≈ 1.0 on GT-as-prediction | Golden SPEED 0.0093 reproduced (byte-identical); bop_toolkit GT-as-prediction AR = **1.0** (VSD via EGL workaround, `dev` and `train`). Image 1.60 with `libegl1` done 2026-10-21 (M3 first bullet; AR 1.0 and golden 0.0093 without the workaround) |
| M1 Refactor and safety net | W2 | `[x]` | SPEED golden diff = 0, pytest green, smoke train OK | Golden byte-identical; 57 tests; smoke val 0.1317; kill/resume continues exactly (LR, losses); done 2026-10-20, a week early |
| M2 BOP data loader (LM-O) | W3–4 | `[x]` | Reprojection and label round-trip tests pass, throughput OK | Started 2026-10-21. Index done: train 320,959 rows (LM PBR 0–47), test 1,445 (= BOP19 targets). `BOPDataset.__getitem__`, reprojection and label round-trip tests and visual check done. Loader 730 samples/s = 2.8× GPU (257). **Closed 2026-10-21**; the optional roll augmentation is deferred |
| M3 Single-object baseline B1 | W4–5 | `[x]` | B1 metrics logged in §9 | Started 2026-10-21, **closed 2026-10-22**. `train.py --dataset LMO` works; overfit check passed (0.43 mm, 0.017°); driller B1 (`lmo-m3-b1-001`): ADD@0.1d 0.87 (LM real val) / 0.77 (LM-O test), rotation 1.7° / 1.6°, translation 13 / 19 mm. Ablations at 384 px / 150 epochs (5 parallel jobs + 2 repeats for noise): `site` depth confirmed (legacy worse), decoupled loss = no effect, **`domain_gap` aug best on real images (LM-O test ADD@0.1d 0.815 vs 0.76–0.77)** |
| M4 Shared 8-object model | W6–7 | `[x]` | ≥ 90 % of B1 on B1's object; no object at 0 ADD | Started 2026-10-22. Bullet 1 done 2026-10-25: 6 parallel 8-object runs. Best `dg4x` (domain_gap, 4× compute): LM real val mean ADD@0.1d 0.663, driller 0.905 (B1 0.870), lowest duck 0.29. `domain_gap` ≫ `standard`; 2× > 1× clearly, 4× ≈ 2× (+0.008). Exit criterion met on val; LM-O test at the end of the milestone. Config confirmed 2026-10-25: domain_gap, 2× for ablations, 4× for final models. Bullet 2 (per-object table): val part done 2026-10-25 (`bop_evaluate.py --obj_ids`, `table_<split>.md`); LM-O test at the end of the milestone. Bullet 3 (class tokens) done 2026-10-26: at domain_gap 2×, 2 seeds each, mean ADD heads 0.660, **token 0.675**, both 0.661; token slightly best (~2× seed spread), a + b = a. Final runs done 2026-10-28: `tok4x` 0.672 / `tok4x2` 0.654 ADD, MSSD 0.809 / 0.797 (≈ `dg4x` and `tok2x`, within noise); **final model `tok4x`** (user, 2026-10-28). **LM-O test (GT boxes): mean ADD(-S)@0.1d 0.511, MSSD 0.653, MSPD 0.891; driller 0.805 = 105 % of B1, lowest duck 0.189 → exit met, M4 closed 2026-10-28.** Eggbox collapses on test (0.43, occlusion) |
| M5 Symmetries | W7–8 | `[x]` | Eggbox/glue improve, others unchanged | Both code bullets done (`--symmetric_rotation_loss`, GPU val metrics + selection on (AR_MSSD+AR_MSPD)/2). Exit jobs `symtok2x`/`symtok2x2` done 2026-10-27: **no measurable gain** (glue MSSD +0.025, eggbox −0.011, means within seed noise). **Kept on by default for BOP from now on (user, 2026-10-27).** Closed without a clear gain |
| M6 Official BOP eval (GT boxes) | W8–9 | `[x]` | Official LM-O AR (oracle) | **Closed 2026-10-28.** Official BOP19 eval of `tok4x` with GT boxes: **LM-O test AR 0.689** (VSD 0.514, MSSD 0.660, MSPD 0.892, 53 ms/image); LM real val AR 0.835. Own MSSD/MSPD = official to ≤ 0.001 (exit ±0.01). BOP CSV `fastposevitmo_lmo-test.csv` from `bop_evaluate.py` with real `time`. DDP: code + tests, N = 1 unchanged; speed-ups 2026-10-28: 1 GPU 221 → 332 samples/s, 2 GPUs 492 → 548 (1.65×; table in M6 and §9); 1 GPU stays the default (user). Optional, not done: 1 vs 2-GPU AR at equal epochs |
| M7 Real detections | W9–11 | `[x]` | First BOP-comparable LM-O AR | Started 2026-11-02. Default detections (GDRNPPDet) downloaded; they match bbox_obj slightly better than bbox_visib (IoU 0.824 vs 0.811, partly amodal); `tok4x` loses 0.05 ADD on amodal GT boxes (val_pbr). **First comparable number: LM-O test AR 0.668 with default detections** (VSD 0.491, MSSD 0.641, MSPD 0.873; GT boxes 0.689); exit criterion met. Crop-source ablation done 2026-11-03 (options 1–3 × 2 seeds, 2×): with detections all within 0.01 AR (0.646–0.655), option 1 (bbox_visib) marginally best, so **kept**; final model stays `tok4x` |
| M8 YCB-V | W11–12 | `[x]` | YCB-V AR + per-object table | Started 2026-11-03. Data report to Pierre handed to Pablo (4 issues, checked against the official HF file lists). Tree + index done: train 626,520 PBR rows, val = PBR scenes 48–49 (no real val), test 4,123 targets; 11 new tests pass. `--dataset YCBV` in train/eval done; 21-object overfit passes. Symmetries with translation offsets done (clamps 19/20 now 1.000 in the overfit check). 4 training jobs done (2026-11-05/06, deleted). **YCB-V test AR 0.702 with default detections** (0.707 with GT boxes), final model `tok4x2` (best on val_pbr). 4× beats 2× on YCB-V (+0.013 AR with detections, +0.03 MSSD on val_pbr), so YCB-V runs keep 4×. Weak objects 16/17: a depth bias of real YCB-V (all objects too far, median +2.4 %; GT fits the depth, objects look smaller in RGB than GT predicts), not fixable without real training data: **accepted as a PBR-only result (user, 2026-11-07)**, `train_real` left as open issue D8. **Closed 2026-11-07** |
| M9 Jetson / deployment | W12–13 | `[x]` | Latency vs number of objects | **Jetson dropped for the semester project (supervisor, 2026-11-18): the A100 latency below is kept as context, the report states the missing Jetson numbers as a limitation.** Started 2026-11-08 (A100 only, user). Bullet 1 done: ONNX export with dynamic batch (`optimization/bop_export.py`), ONNX Runtime (CPU) vs PyTorch fp32 max diff **3.0e-6** at batch 1/4/8/16 (< 1e-3). Bullet 2 done: TensorRT fp16 engine (profile 1/8/16), **LM-O AR 0.6685 = PyTorch 0.6685, YCB-V 0.7017 vs 0.7021** with detections (< 0.005); SPEED engine builds again. Bullet 3 done: latency table (A100): TensorRT fp16 pipeline **5.3 / 9.3 / 14.2 / 24.0 ms for N = 1 / 4 / 8 / 16** (network alone 1.8 / 3.5 / 5.7 / 11.0); with the BOP default detector (82 ms) ~10 FPS → exit met; stretch (own multi-class LW-DETR detector): 3 training jobs running since 2026-11-08 ~11:00 |
| M10 Write-up (+ T-LESS stretch) | W13–14 | `[~]` | Report and final tables | Per-object models (8 single-object LM-O runs, 2026-11-08): **AR 0.672 with detections vs shared `tok4x` 0.668** (GT boxes 0.688 vs 0.689), within noise: keep the shared model. Write-up draft `docs/BOP.md` + README BOP section (2026-11-08). 224 px: AR 0.643 / 0.643 (−0.023). Paper §3 ablations done 2026-11-11 (absrot −0.11, scratch −0.14, legacy −0.04, bbox0 −0.03, bbox20 ≈). LaTeX draft `paper/` with figures (2026-11-11); leaderboard check: not best on LM-O, PViT-6D prior art (positioning with the supervisor). Augmentation study done 2026-11-14 (`domain_gap` p = 0.5 kept: best 0.655; none 0.627, color 0.645, blur 0.629, noise 0.627, all three 0.642, p = 0.8 0.636); box jitter study done 2026-11-15 (test AR, mean of 2 seeds: 5 % 0.650, 10 % 0.655, 15 % 0.653, 30 % 0.644; 10 % kept). Paper split into `paper/main.tex` + `paper/supplementary.tex` (2026-11-14). **T-LESS done 2026-11-15** (`tok4x` / `tok4x2`, 4×, PBR only): **AR 0.762 / 0.762 with default detections** (GT boxes 0.813 / 0.812), RGB-D 0.858 / 0.859. M11 phase A done (context −0.03, occlusion and translation weight within noise: nothing adopted); baseline re-trained on the current code 0.652 (original 0.655); single-object models, seed 1: 0.666 (mean 0.669, shared 0.666). **2026-11-18: the write-up becomes the EPFL report (due 2027-01-01) → M20** |
| M11 Improve first | W5 | `[x]` | Changes chosen on validation | **Closed 2026-11-18.** Phase A: nothing adopted under the old rule. **`tlw` + `occl` adopted under the new rule (2-seed means at 4×: test 0.668 vs 0.666, validation 0.886 vs 0.883).** B1 → M15, B3 → M17, B4 → M20; B2 8×: seed a done (test 0.6728, noisy validation 0.8861), seed b `lmo-m13-base8x2-001` running since 2026-11-20; C RGB-D done (C8 with the finals) |
| M15 Backbone and resolution | W5–W7 | `[~]` | Best backbone / input size by the rule | **2026-11-20 (adoption rule, `tlwocc{,2}` 2× baseline: noisy validation 0.8825, test 0.6606):** ViT-B/512 (2 seeds) **0.8824 / 0.6605: tied, not adopted**; ViT-L/512 seed b alone 0.8902 / 0.6814 (+0.008 / +0.021; PBR validation 0.942 vs 0.909), **seed a still running (epoch 105 of 150, preempted)**; ViT-B/640 and ViT-H/518 still running |
| M16 One network per object | W5–W7 | `[x]` | Per-object vs shared by the rule, 3 datasets | **Closed 2026-11-20: not adopted on any dataset** (noisy validation and official test AR, 2-seed means vs the shared `tok4x{,2}`): LM-O 0.879 / 0.669 vs 0.883 / 0.666; YCB-V 0.674 / 0.713 vs 0.691 / 0.698 (test better, validation worse); T-LESS 0.845 / 0.736 vs 0.878 / 0.762 (clearly worse on both) |
| M17 PBR + real training | W5–W7 | `[~]` | PBR + real vs PBR only by the rule, 3 datasets | **2026-11-20 (adopt for the PBR + real category, always labelled):** YCB-V 2× noisy validation on held-out real videos 0.752 vs 0.676 (`tok2x{,2}`), test 0.764 vs 0.685; LM-O 2× test 0.681 vs 0.661 (`tlwocc{,2}`), validation 0.972 vs 0.883 but biased (neighbouring real views in training). T-LESS running (epochs 91–131 of 150) |
| M18 Multi-head: DR + masks + PnP | W6–W7 | `[~]` | DR / PnP / combined, AR and time | New architecture (supervisor): one ViT, heads for direct regression, visible mask and dense object coordinates; PnP at test time. **Code done 2026-11-20 (M18.1–M18.5, 33 new tests, full suite green), no training yet:** dense targets, `--dense_heads`, `src/pnp.py` (4 pose modes), `bop_evaluate.py --pose_mode`, `scripts/m18_dense.sh`. Open: M18.6 (symmetric XYZ labels, rendered targets for real images, loss weights) |
| M19 Render-and-compare refinement | W6–W7 | `[~]` | AR gain of refinement | **R1 done 2026-11-20: CosyPose's public PBR refiner gives no gain on our poses** (LM-O 0.669 → 0.616 after 1 iteration, YCB-V 0.694 → 0.693, T-LESS 0.762 → 0.743; it does not refine errors below ~3° / 15 mm). R2 (our own refiner) needs a decision |
| M20 Final models and EPFL report | W7–W11 | `[ ]` | Report sent by 2027-01-01 | Decision gate 2026-12-04, finals W8–W9, report on `paper/` |

---

## 2. Baselines

### 2.1 SPEED (existing best model, non-regression reference)
Pretrained HF weights `weights/vit_b_16_384.pth` (ViT-B/16 at 384 px), evaluated on `SPEED_FIXED` test (1,200 images) with the boxes in `speed_bbox_annotations.json`:

| Metric | Value |
|---|---|
| Translation error E_T | 0.0221 m |
| Rotation error E_R | 0.0073 rad (0.418°) |
| Relative translation | 0.0020 |
| **SPEED score** (rel. trans + rot) | **0.0093** |

- Reproduced twice: `$NAS_HOME/results/eval-pretrained-vit-b-16-384/` (2026-09-30) and `$NAS_HOME/results/eval-rcp-check/` (2026-10-13).
  `results/eval-rcp-check/results.json` holds per-sample predictions and is the **golden file** for M1 (eval runs in fp32, ~15 s).
- Matches the paper (Ancey et al., WACV 2026, [arXiv 2512.09792](https://arxiv.org/abs/2512.09792)), Table 2.
- Other paper numbers:
  - Best SPEED variant: ViT-L-512/16, 0.0182 m / 0.338°. Its weights are not public.
  - ViT-B-224/16: 0.0231 m / 0.447°.
  - SPEED+ (ViT-B-384): lightbox 0.270 m / 12.26°, sunlamp 0.400 m / 18.28°.
- Training cost: the full 310-epoch run takes about 9 h. 310 = 10+20+40+80+160, the warm-restart schedule with T0 = 10 and Tmult = 2.

### 2.2 BOP
- **No FastPose-ViT number exists on BOP yet.** The plan creates them in order:
  - B1 (M3): the original architecture on one object.
  - M6: the multi-object AR with GT boxes (oracle).
  - M7: the AR with real detections, the first number comparable to BOP entries.
- Reference numbers for RGB-only methods. **Verify them on the [BOP leaderboard](https://bop.felk.cvut.cz/leaderboards/), including whether refinement is used, before quoting.**

  | Method | LM-O | T-LESS | YCB-V | Source |
  |---|---|---|---|---|
  | GDRNPP-PBR-RGB-MModel | 71.3 | 79.6 | 71.3 | BOP Challenge 2022 paper ([arXiv 2302.13075](https://arxiv.org/abs/2302.13075)) |
  | GDR-Net | 67.2 | 70.8 | – | as reported by a later paper ([arXiv 2607.02360](https://arxiv.org/abs/2607.02360)) |
- **LM-O BOP19 leaderboard, per metric** (read 2026-10-28 from the [BOP leaderboard, LM-O](https://bop.felk.cvut.cz/leaderboards/pose-estimation-bop19/lm-o/) through an automatic page extractor: spot-check a value on the page before quoting it). All entries use real detections. "Training data" is taken from the entry name ("?" = not stated).
  - RGB only (like us):
    | Method | Training data | AR | AR_VSD | AR_MSSD | AR_MSPD | time (s/img) |
    |---|---|---|---|---|---|---|
    | IL-Pose | ? | 0.772 | 0.621 | 0.794 | 0.900 | – |
    | HccePose (FCOS) | ? | 0.768 | 0.615 | 0.787 | 0.902 | – |
    | ZebraPoseSAT-EffnetB4 (PBR only, DefaultDetections-2023) | PBR | 0.729 | 0.562 | 0.736 | 0.888 | 0.250 |
    | GDRNPP-PBR-RGB-MModel | PBR | 0.713 | 0.549 | 0.701 | 0.887 | 0.206 |
    | GDRNPP-PBRReal-RGB-SModel | PBR + real | 0.686 | 0.518 | 0.664 | 0.875 | 0.092 |
    | SurfEmb-PBR-RGB | PBR | 0.663 | 0.497 | 0.640 | 0.851 | 11.8 |
    | CosyPose-ECCV20-PBR-1VIEW | PBR | 0.633 | 0.480 | 0.606 | 0.812 | 0.550 |
    | CDPNv2_BOP20 (PBR-only & RGB-only) | PBR | 0.624 | 0.445 | 0.612 | 0.815 | 0.163 |
    | EPOS-BOP20-PBR | PBR | 0.547 | 0.389 | 0.501 | 0.750 | 0.468 |
    | DPODv2 (synthetic, RGB, PnP) | synthetic | 0.584 | 0.432 | 0.560 | 0.761 | 0.274 |
    | **Ours, M4 `tok4x`, default detections (official eval, 2026-11-02)** | PBR | **0.668** | 0.491 | 0.641 | 0.873 | 0.134 |
    | Ours, M4 `tok4x`, GT boxes (oracle, official eval, 2026-10-28) | PBR | 0.689 | 0.514 | 0.660 | 0.892 | 0.053 |
  - RGB-D, for reference only (they use depth): the best reach AR 0.80–0.86 (e.g. Flose 0.861: VSD 0.817, MSSD 0.875, MSPD 0.891; GDRNPP-PBR-RGBD-MModel 0.775: 0.629 / 0.822 / 0.872). Depth mostly fixes the distance error, hence MSSD 0.82–0.88.
  - GDR-Net (not GDRNPP) and PoseCNN are not on this page.
  - Reading (2026-10-28): our MSPD (2D alignment) is at the level of the best RGB methods; our MSSD (3D, dominated by the depth error of the directly regressed translation) is between SurfEmb and GDRNPP. Ours uses GT boxes, so it is optimistic until M7. RGB methods have AR_VSD ~0.15 below AR_MSSD, so a rough guess for our official AR is the low 0.6s; M6 measures it.

---

## 3. Repository map (what matters for this task)

| File / folder | Today | Change for multi-object BOP |
|---|---|---|
| `src/camera.py` | One global SPEED camera (1920×1200, fixed K) | Per-sample K. The SPEED camera stays as the default |
| `src/utils.py` | Bbox-relative translation (`translation_to_bbox_relative_translation` and inverse) and allocentric rotation (`get_apparent_orientation` / `get_absolute_orientation`), all using the global Camera | Optional `K`, `img_wh`, `depth_mode`, `diameter` args; old path unchanged when `K is None` |
| `src/datasets.py` | `SPEEDDataset`, `DATASET_CONFIGS`, `crop_image`, `SpatialDataAugmenter`, pixel/style aug pipelines | Reuse pixel augs and `random_bbox_augmentation`; split rotation aug from bbox jitter |
| `src/engine.py` | `train_epoch` / `val_epoch`, batch = `(img, t, R, bbox)`, `CoordinateTransformer` | `unpack_batch` accepting tuple (SPEED) or dict (BOP); per-sample K; per-object head gather; symmetric loss call |
| `src/models.py`, `src/vit_models.py` | `FastPoseViT`, one head (best config: `--no_mlp --merge_outputs`, 9 outputs), optional multiple class tokens | `num_objects` (per-object heads); optional per-object class tokens |
| `src/losses.py`, `src/metrics.py` | 6D relative Frobenius + relative translation; SPEED score | Symmetry-aware 6D loss; ADD(-S), MSSD, MSPD |
| `src/train.py`, `src/evaluate.py`, `src/predict.py` | SPEED-only `--dataset`; keeps the best model by val loss; no resume | BOP datasets; resume; AR-based model selection; BOP19 CSV export |
| `object_detector/` | LW-DETR, single class (`num_classes=1`, label `"satellite"`) | Not needed at first (BOP default detections). Multi-class only as a stretch goal |
| `optimization/` | ONNX/TensorRT with a fixed `(1,3,H,W)` input | Dynamic batch (one crop per object) |
| `other_utils/` | Visualization (SPEED K hardcoded), dataset splitters, `merge_annotations.py` | Reuse projection/overlay code for BOP sanity checks |
| `scripts/*.sh` | SPEED ablation sweeps (backbones, tokens, crop %, augs, optimizer) | Add `scripts/bop_lmo.sh` |
| `tests/` | Import smoke test only (`test_imports.py`) | Geometry, loader, loss and SPEED-golden tests |
| `rcp/`, `docs/RCP_*.md`, `AGENTS.md` | RCP cluster workflow | Unchanged (project notes appended to AGENTS.md) |

**New files planned:**
- `src/bop_dataset.py`: index, crop warp, labels.
- `src/bop_eval.py`: model points, symmetry table, batched ADD(-S)/MSSD/MSPD, BOP19 CSV writer.
- `src/predict_bop.py`: inference on GT boxes or a detections file.
- `scripts/make_bop_tree.sh`: symlinked standard BOP tree.
- `tests/test_rotation_convention.py`, `tests/test_translation_param.py`, `tests/test_bop_dataset.py`, `tests/test_symmetry_loss.py`, `tests/test_speed_golden.py`.

**Reuse rather than rewrite:**
- From `src/`: `get_rotation_matrix`, `rotation_matrix_to_6d`, the R_rel logic in `_compute_orientation_transform`, `random_bbox_augmentation`, the pixel and style augmentation pipelines, `train_epoch` / `val_epoch`, `LossAccumulator`, `RelativeRotation6DFrobeniusLoss`, `compute_metrics`.
- From `bop_toolkit_lib`: `inout.load_ply`, `misc.get_symmetry_transformations`, and `pose_error.mssd` / `mspd` / `adi` as test references.

**`.gitignore` gotchas:**
- ~~`test*` also ignores `tests/test_*.py`~~: the rule was removed in `6b3c430`, so new tests are tracked.
- `*.json` ignores any new JSON config, and also `.claude/settings.json`.
- Ignored files never reach a `train` job's code snapshot.

---

- **Verification audit 2026-11-14, W10 (maintainability):** `TASKS.md` has two lines over 3,000 characters (§10 item 7, one §1 row) and `src/train.py` is 1,495 lines.
  - [x] Split the overlong `TASKS.md` lines **Status 2026-11-14:** longest line 7,999 -> 1,478 characters; the M2–M9 history of §10 moved verbatim to `docs/TASKS_HISTORY.md`. into sub-bullets / a short history pointer without losing content.
  - [x] Move the argument parser of `src/train.py` **Status 2026-11-14:** `src/train.py` 1,495 -> 1,101 lines, parser in `src/train_args.py`; `--help` byte-identical, 175 tests pass, smoke run of the ablation script OK. into `src/train_args.py` (pure move; tests + smoke run must be identical).

## 4. Agent setup

- **Decided 2026-10-20: `AGENTS.md` is the only agent-instructions file; no `CLAUDE.md` is needed.**
- **Start Claude from the repo root inside the `dev` job:** `dev`, then `cd $NAS_HOME/FastPose-ViT && claude`. The 2026-10-20 session was started from `$NAS_HOME` on what looked like the jumphost (no GPU, `RCP_JOB` unset). There, AGENTS.md is only loaded on demand and the `.claude/settings.json` permission rules don't apply.
- **Session routine:** read §1 and §10 → work → update §1, §9, §10 → update AGENTS.md project notes if a convention changed.

---

- **Verification audit 2026-11-14, W9 (security, low risk):** the GitHub token is plain text in `~/.git-credentials` (owner-only file) and a classic `repo` token covers every repository; checkpoints load with full pickle (`torch.load(weights_only=False)`), including the third-party LW-DETR COCO weights.
  - [x] Load untrusted / downloaded files with `weights_only=True` **Status 2026-11-14:** every `weights_only=False` removed; `src/safe_load.py` (weights_only + data-only allowlist: Namespace, numpy RNG types) loads training checkpoints, detector files and the third-party COCO weights; a test checks that a malicious pickle is refused (code does not run); the COCO files' sha256 is in `$NAS_HOME/weights/final/MANIFEST.json`. (COCO weights: check they load, record the sha256), use a small checkpoint loader that tries `weights_only=True` first.
  - [x] Token: user decision 2026-11-14: keep the current classic token (accepted risk: it covers all repositories; keep `~/.git-credentials` owner-only and revoke the token when the project ends) **Status 2026-11-14:** open, user action., replace the classic token by a fine-grained one limited to `PabloSarro/FastPose-ViT-MO` (Contents read / write) and a short expiry; steps given in the chat, tracked here.
  - [x] Check that no secret is tracked **Status 2026-11-14:** tracked files and the whole git history scanned (GitHub / Hugging Face tokens, AWS keys, private keys, passwords): nothing found; `rcp/project.env` holds cluster settings only. in the repository (grep).

## 5. Data inventory and known issues

BOP root: `$DATA_DIR/BOP_Datasets` (owned by Pierre, read-only for us). The scan on 2026-10-20 found all folders readable from the jumphost.

| Dataset | State | Notes |
|---|---|---|
| `lm` | Complete | `lm_train_pbr/train_pbr` (50 scenes × 1,000 imgs), `lm_test_bop19/test` (15 scenes), models / models_eval (15 objects) |
| `lmo` | Complete | `lmo_test_bop19/test/000002` (200 imgs, 1,445 targets), `lmo_base/lmo/test_targets_bop19.json`, `lmo_models/{models,models_eval}`. Objects 1, 5, 6, 8, 9, 10, 11, 12. 10 (eggbox) and 11 (glue) have one discrete symmetry each. No own train_pbr: BOP practice is LM's PBR only |
| `ycbv` | Mostly complete | train_pbr (50), train_synt (80), test_bop19 (12 scenes / 900 imgs). **`train_real` broken**: `ycbv_train_real.z01` missing, only 3 near-empty scenes. Two cameras (uw, cmu). 21 objects; 13 and 18 continuous symmetry |
| `tless` | Complete | train_pbr, train_primesense, test_primesense_bop19. 30 objects, 27 symmetric |
| `ipd` | Extracted | BOP-Industrial, multi-camera, 3840×2160. Test GT hidden. Out of scope for now |
| `hb` | **Empty** | Its zips were downloaded into `xyzibd/` |
| `xyzibd` | **Zips only** | `train_pbr.z01`, models and test_all missing |

**Layout quirk:**
- Splits are nested, e.g. `lm/lm_train_pbr/train_pbr/000000` and `lmo/lmo_base/lmo/camera.json`.
- `bop_toolkit` expects the standard layout, so build a symlinked tree in `$NAS_HOME/datasets/bop/` (M2). Do not copy or modify anything in `$DATA_DIR`.

**Measured on LM PBR scene 000000:**
- 15,000 instances, of which 8,000 belong to LM-O objects and 6,866 have `visib_fract ≥ 0.1`.
- That is about **343k training crops** over 50 scenes, roughly 43k per object.
- One camera: fx 572.4, fy 573.6, **cx 325.3, cy 242.0**, so the principal point is not the image centre. `depth_scale` is 0.1.
- LM-O model diameters range from 102 to 262 mm.

**Splits for LM-O:**
- **train:** LM `train_pbr` scenes 0–47.
- **val:** PBR scenes 48–49 (synthetic), plus LM `test_bop19` scenes 1, 5, 6, 8, 9, 10, 11, 12 (real; each annotated only for its own object). Scene 2 is the LM-O test sequence, so leaving it out means no leak.
- **test:** LM-O `test_bop19`, used only at milestones and never for early stopping or model selection.

**Report to Pierre (dataset owner):** deferred on 2026-10-20. None of these issues affects LM-O, so send them before starting YCB-V (M8). Re-checked 2026-11-03 (all still present, and against the official file lists at `huggingface.co/datasets/bop-benchmark/<ds>`: the only missing split parts are `ycbv_train_real.z01` and `xyzibd_train_pbr.z01`; IPD's are all there). Message (items 1–4; the IPD item left out by Pablo) handed to Pablo, who sends it; the draft file was removed.
- [ ] `hb/` is empty; the hb zips are in `xyzibd/`.
- [ ] xyzibd is missing `xyzibd_train_pbr.z01`, `xyzibd_models.zip` and `xyzibd_test_all.zip`.
- [ ] `ycbv_train_real.z01` is missing, so train_real cannot be extracted.
- [ ] There is a stray empty folder literally named `*`.
- [ ] IPD `test/000000` and `test/000004` contain `.ipynb_checkpoints/scene_gt_cam1-checkpoint.json` (test GT). **Never use these files.**

`$NAS_HOME/BOP_Datasets` was a broken self-referencing symlink. It was repaired on 2026-10-20 to point to `$DATA_DIR/BOP_Datasets`.

---

## 6. Design notes (verified in the code, 2026-10-20)

1. **Rotation convention: no transpose needed.**
   - `quaternion_to_rotation_matrix(q_vbs2tango)` (`src/utils.py`) is the standard Hamilton matrix, while ESA's `quat2dcm` (`other_utils/visualization.py`) is its transpose. `project_axes` computes `p_cam = quat2dcm(q)ᵀ·p_body + r`.
   - So the SPEED pipeline already regresses **R model→camera**, exactly BOP's `cam_R_m2c` (`x_c = R·x_m + t`).
   - Use BOP rotations as they are, reshaped row-major to 3×3. BOP translations are in **mm**, SPEED's in metres.
   - The allocentric conversion `R_relᵀ·R` (`get_apparent_orientation`) is valid for BOP too.
2. **Depth parameterisation must change for BOP.**
   - The current `U_z = t_z·(w+h)/(2H)` is about 50–300 on BOP (mm, H = 480), while `U_x`, `U_y` are about ±0.5.
   - `RelativeTranslationLoss` divides by ‖U‖ ≈ U_z, so the 2D centre would be almost ignored.
   - Add a `site` mode `U_z = t_z·s / (f̄·D_obj)` with s = (w+h)/2, f̄ = (fx+fy)/2 and D_obj the object diameter. It is dimensionless, about 1, and independent of camera and object size.
   - Keep the legacy formula bit-identical when `K is None` (SPEED).
   - Also consider a decoupled loss `L1(U_x,U_y) + λ·L1(U_z)` as an ablation.
3. **The spatial rotation augmentation is wrong for BOP.**
   - `R_change = K⁻¹·M·K` is a pure camera roll only if the rotation centre equals the principal point and fx = fy.
   - LM has a principal point offset of about (5.8, 2.6) px and fx ≠ fy, which gives about 20 mm label error at 180°.
   - Turn it off for BOP. Split a new `--no_rotation_augmentation` from bbox jitter: today `--no_spatial_augmentation` disables both.
   - Exact alternative for later: warp with `K·Rz(θ)·K⁻¹` and use labels `t' = Rz·t`, `R' = Rz·R`.
   - Also drop `A.OpticalDistortion` for BOP, since it is a geometric warp that makes labels noisy.
   - Pixel augmentation should be **on** for the PBR→real gap (SPEED's best config used `--no_pixel_augmentation`; don't copy that).
4. **Object conditioning: start with per-object heads.**
   - `Linear(768, 9·n_obj)`, then select the 9 outputs of the sample's object in the engine.
   - With n_obj = 1 it is identical to today, so SPEED weights load unchanged and the ONNX export needs no extra input.
   - Ablation: per-object class tokens (`src/vit_models.py`). This needs an `obj_idx` model input.
   - One model per object is only used as the M3 baseline (it does not scale and can't batch crops of different objects).
5. **Symmetries.**
   - Loss = min over k of the 6D loss between prediction and `R_gt·S_k`, with `S_k` from `models_info.json` via `bop_toolkit_lib.misc.get_symmetry_transformations`.
   - Symmetries multiply on the right and the allocentric change on the left, so this works directly on allocentric targets.
   - Continuous symmetries (YCB-V objects 13 and 18, T-LESS) are discretised, or use an axis-based loss.
6. **Model selection.** `train.py` keeps the lowest symmetry-unaware val loss. For BOP, select on `(AR_MSSD + AR_MSPD)/2` computed on the val split.
7. **Training budget.**
   - Use "virtual epochs" of about 25k samples, so the existing warm-restart schedule (70/150/310 epochs) still applies.
   - The LR is auto-scaled linearly from base batch 4. Check this with Muon at batch 32.
   - Add checkpoint/resume: jobs are limited to 12 h and can be preempted, and today only `model.pth` (best) is saved.
8. **Deployment.** Multi-object means one crop per object, so ONNX and TensorRT need a **dynamic batch** (today the export is fixed at `(1,3,H,W)`).
9. **Pixel convention (found in M2, 2026-10-21).** BOP pixel i is centred at i + 0.5 in K coordinates, and `[x, y, w, h]` spans [x, x+w+1]. Convert every BOP or detector box with `bop_bbox_to_xyxy`; `crop_warp_matrix` handles the half-pixel offsets.
10. **Risks.**
   - VSD needs a headless renderer (EGL or OSMesa in the container). It is tested in M0.
   - NAS I/O: each PBR image is decoded about 7 times per pass.
   - PBR→real domain gap.
   - LR scaling at large batch.
   - Test leakage if LM-O test is used for any decision.

---

## 7. Milestones and sub-milestones

### M0 — Setup and data access (W1)
- [x] Repair `$NAS_HOME/BOP_Datasets` symlink (done 2026-10-20; `ls $NAS_HOME/BOP_Datasets/lmo` works).
- [x] Create branch `multi-object`; commit `TASKS.md` and the AGENTS.md project notes (done 2026-10-20). Both were already on `main` in `6b3c430`; the branch starts there and this update is its first commit.
- [x] Inside `dev`, create `.venv` with `uv venv --system-site-packages` and install `bop_toolkit` (`uv pip install git+https://github.com/thodan/bop_toolkit`).
  - Dry-run first so it doesn't replace the image's torch or numpy.
  - Check: `.venv/bin/python -c "import torch, bop_toolkit_lib; print(torch.cuda.is_available())"`.
  - **Done 2026-10-20** (job `fastpose-dev`, A100). What was done differs from the plan:
    - `uv` is not in the image, so: `python3 -m venv --system-site-packages .venv`, then `.venv/bin/pip install uv` (so `.venv/bin/uv` exists).
    - **Use `.venv/bin/pip`, not uv, for installs into this venv.** uv ignores the system site-packages and would reinstall duplicates of numpy, scipy and Pillow.
    - The bop_toolkit dry run would have downgraded numpy 2.3.5 → 1.26.4 (it pins `numpy<2`) and added `opencv-python` next to the image's `opencv-python-headless`. Instead:
      - Cloned it to `$NAS_HOME/bop_toolkit` (commit `cea62d6`, outside the repo). The clone is needed for `scripts/eval_bop19_pose.py`, which a pip install doesn't ship.
      - `.venv/bin/pip install --no-deps -e ../bop_toolkit`.
      - Added only the missing deps: pypng, webdataset, imageio, scikit-image, vispy, matplotlib.
    - Result: torch 2.9.1+cu130 (CUDA True), numpy 2.3.5, cv2 4.13.0 and Pillow 10.4.0, all unchanged. `pip check` only reports the two intentional pin violations (numpy<2, opencv-python).
    - Smoke test under numpy 2: `inout.load_json` on LM-O `models_info.json`, `misc.get_symmetry_transformations` (eggbox → 2), and `pose_error.adi/mssd/mspd` (identity → 0). All OK. The renderer (`bop_toolkit_lib/rendering`) is tested in the VSD step below.
- [x] Verify BOP read access **inside a `train` job**, not just `dev`, with a tiny job that reads one PBR jpg, `scene_gt.json` and one `models_eval` ply. Commit `1c08f08` added the supplemental groups for this.
  - **Done 2026-10-20 with `lmo-m0-access-004`** (1 GPU, 2 CPU, 8 Gi): `bash rcp/train.sh lmo-m0-access-004 bash -c "id && .venv/bin/python -W ignore scripts/check_bop_access.py --dataset lmo"`.
    - Inside the job: groups `30257,11166,79706`. It read the RGB image (480×640×3), `scene_gt.json` (1,000 images, 15 instances in image 0) and `obj_000001.ply` (2,825 points). It printed `BOP ACCESS OK (lmo)` and exited 0.
    - `scripts/check_bop_access.py` also has a `--dataset ycbv` option, for M8.
    - `lmo-m0-access-003` (submitted by the user from the jumphost, `id; ls $DATA_DIR/BOP_Datasets/lmo`) also succeeded.
  - **Lessons (RCP quirks):**
    - Jobs submitted at 14:51 UTC only appeared in `runai list` around 15:00, after about 9 min. Meanwhile `wait.sh` returned 2 ("no such job").
      - **This was a temporary delay on the cluster side, not the norm.** Re-test at 15:59 UTC with 5 tiny jobs (`lmo-m0-lag-001…005`, 1 GPU, 1 CPU, 1 Gi, `echo hi`): all appeared in `runai list` within **7–12 s** of submission, as Pending.
      - They started running 0.5–2.3 min after submission, waiting for a GPU, and all exited 0. All were stopped afterwards.
      - Rule: if a job isn't listed within ~1 min, re-check for a few minutes before concluding it failed. `$RUN_DIR/log.txt` ("exit code" line) is the reliable signal that it ran.
    - **Don't put multi-line inline Python in `train.sh` commands.** runai strips the inner `"`, which gave `lmo-m0-access-002` a SyntaxError and a restart loop. Use a script file instead.
    - **`GPUS=0` jobs went to a V100 node** and were still pulling the image after several minutes (`lmo-m0-access-001`, `lmo-m0-cpuprobe-001`). For short jobs, 1 GPU with few CPUs is faster.
  - All these test jobs were stopped, and their run folders were kept. The exception is `lmo-m0-subdebug-001`, submitted by hand: `stop.sh` refuses it and `runai delete` is blocked by permissions, so the user has to delete it.
- [x] Add `.gitignore` exceptions (`!tests/**`, and any config JSON you add); check `pytest tests/` runs.
  - Found on 2026-10-20: `6b3c430` already removed the `test*` rule (so `tests/test_*.py` is no longer hidden), as well as the rules for `.pytest_cache`, `results`, `runs`, `checkpoints`, `temp` and `.ruff_cache`. It also added `AGENTS.md`, which has no effect because the file is tracked. `*.json` is still ignored.
  - The removals are intentional: these folders are empty for now. Add ignore rules back (e.g. `.pytest_cache`) once they start filling up.
  - **Done 2026-10-20:**
    - In `.gitignore`, added `!tests/**/*.json`, so JSON fixtures in `tests/` are tracked despite `*.json`. A blanket `!tests/**` would also have un-ignored `tests/__pycache__/`, so it wasn't used.
    - Added `.pytest_cache/` back, since running pytest now creates it. This replaces the dangling comment "Ignore all test folders except tests".
    - No config JSON exists yet. When adding one (e.g. `configs/`), add a `!configs/**/*.json` exception.
    - `.claude/settings.json` is still ignored (left as is).
    - Added `tests/test_imports.py`, an import smoke test for 10 `src.*` modules and `bop_toolkit_lib.{inout,misc,pose_error}`. With 0 tests, pytest exits 5 ("no tests ran").
    - `.venv/bin/python -m pytest tests/` → **13 passed in 6.2 s**, with no `PYTHONPATH` needed (`tests/` is a package, so pytest puts the repo root on `sys.path`).
    - Verified with `git check-ignore`: `tests/**/x.json` is tracked; `tests/__pycache__/`, `src/__pycache__/` and `.pytest_cache/` are ignored.
- [x] Golden SPEED eval with `.venv/bin/python` (command in `docs/RCP_WORKFLOW.md` §5). It must print score 0.0093; diff its `results.json` against `$NAS_HOME/results/eval-rcp-check/results.json`.
  - **Done 2026-10-20** in `dev`, with the §5 command, `.venv/bin/python` and output in `$NAS_HOME/results/eval-m0-venv-golden/`:
    - E_T 0.0221, E_R 0.0073, relative translation 0.0020, **SPEED score 0.0093**, in about 25 s wall time.
    - `results.json` is **byte-identical** to the golden `eval-rcp-check/results.json` (1,201 entries, 4,805 leaves, max abs diff 0.0).
  - Side note: the ViT builder downloads torchvision's ImageNet `vit_b_16_swag-9ac1b537.pth` (331 MB) into the container-local `TORCH_HOME=/app/tmp/torch_cache` before our weights overwrite it. That costs about 2 s per new container and doesn't affect the result. It could be skipped in M1 when full weights are loaded.
- [x] bop_toolkit sanity check:
  - Build `scripts/make_bop_tree.sh` (symlinked tree in `$NAS_HOME/datasets/bop/`).
  - Write a **GT-as-prediction** CSV for LM-O and run `eval_bop19_pose.py`. Expect AR ≈ 1.0.
  - This tests the VSD renderer. If EGL fails, try `NVIDIA_DRIVER_CAPABILITIES=all`, or OSMesa in a new image tag.
  - **Done 2026-10-20 in `dev`: AR = 1.0** (VSD 1.0, MSSD 1.0, MSPD 1.0, all thresholds), 53 s with 8 workers. Results in `$NAS_HOME/results/bop-m0-gtpred-wrapper/`.
  - `scripts/make_bop_tree.sh [DS ...]` (default `lmo lm`) builds `$NAS_HOME/datasets/bop/<ds>/` from directory-level symlinks.
    - Example: `lmo/test → lmo_test_bop19/test`, `lmo/models_eval`, `lm/train_pbr`, plus the `<ds>_base/<ds>/*` files (camera.json, test_targets_bop19.json).
    - `lmo_test_all/test` collides with `test`, so it is linked as `test_all`. The same holds for LM.
    - Safe to re-run (`ln -sfn`). Nothing is written into `$DATA_DIR`, checked.
  - `scripts/bop_gt_as_prediction.py --dataset lmo --out_dir DIR` writes `gtpred_lmo-test.csv`: 1,445 targets → 1,445 estimates, t in mm.
  - **`scripts/bop_eval.sh RESULTS_DIR CSV`** is the official eval wrapper; use it from now on. Scores go to `RESULTS_DIR/eval/<method>_<ds>-<split>/scores_bop19.json`.
  - **EGL problem and workaround:**
    - The first VSD run failed: the image has `libglvnd0`, Mesa and the NVIDIA EGL libs (`libEGL_nvidia`, `libnvidia-eglcore`, `10_nvidia.json`), but **not `libEGL.so.1` (Ubuntu package `libegl1`)**.
    - `NVIDIA_DRIVER_CAPABILITIES=all` would not help: it only mounts vendor libraries.
    - Fix without root: downloaded `libegl1_1.7.0-1build1_amd64.deb` with `apt-get download`, extracted it to **`$NAS_HOME/.local/lib/egl/`**, and pointed `EGL_LIBRARY` at it (vispy uses `find_library`, which ignores `LD_LIBRARY_PATH`).
    - Rendering then uses the A100 (`GL_RENDERER = NVIDIA A100-SXM4-80GB`).
    - `bop_eval.sh` sets all of this up automatically, and skips it once the image has `libEGL.so.1`.
    - **Also works inside a `train` job**, checked 2026-10-20 with `lmo-m0-bopeval-001` (1 GPU, 8 CPU, 16 Gi, node gpu007, still image 1.59 + workaround).
      - GT-as-prediction gives AR 1.0 (VSD/MSSD/MSPD 1.0; every object and scene recall 1.000). The eval took 63 s.
      - Submitted 21:13:39, started 21:19:58: about 6 min of queueing/image pull on a new node. The job was then stopped.
      - The command pattern was `bash rcp/train.sh NAME bash -c 'export BOP_PATH=$NAS_HOME/datasets/bop && ... && bash scripts/bop_eval.sh $RUN_DIR file.csv'`.
  - Other gotchas:
    - `eval_bop19_pose.py` starts its sub-steps with a bare `python`, which isn't on PATH here, so `.venv/bin` must come first on PATH. The wrapper does this.
    - `/usr/bin/time` is not in the image.
    - MSSD/MSPD alone (no renderer) were also checked: 1.0 (`results/bop-m0-gtpred/eval_novsd`).
- **Exit:** golden SPEED 0.0093 reproduced, and toolkit AR ≈ 1.0.

### M1 — Safety net and refactor, no behaviour change (W2)
- [x] `tests/test_rotation_convention.py`:
  - `quat2dcm(q)ᵀ == quaternion_to_rotation_matrix(q)` on random q.
  - ego → allo → ego round trip.
  - `R_rel(t)·ẑ = t̂`.
  - Roll equivariance.
  - **Done 2026-10-20: 9 tests pass, all in float64 with tolerance 1e-9.**
    - `quat2dcm(q)ᵀ == quaternion_to_rotation_matrix(q)` on 256 random quaternions.
    - Extra: ESA's `project_axes` equals `K(R·p + t)` with `R = quaternion_to_rotation_matrix(q)`. So SPEED's R is model→camera, the same as BOP `cam_R_m2c`, and §6.1 is now tested. A transposed R would be off by 89 px, versus 3e-13 px.
    - `R_rel(t)·ẑ = t̂` on 256 translations, including one exactly on the optical axis (→ identity). The axis of R_rel is ⟂ ẑ, so it is the smallest rotation and adds no roll.
    - Ego → allo → ego and allo → ego → allo round trips (`get_apparent_orientation` / `get_absolute_orientation`), plus the explicit formula `allo = R_rel(t)ᵀ·R`.
    - Roll equivariance for 4 angles: `t' = Rz·t`, `R' = Rz·R` ⇒ `allo' = Rz·allo`.
  - `tests/` now has 22 tests (13 import + 9 rotation), all passing in 7 s. The new file is formatted with ruff (line length 99).
- [x] `tests/test_translation_param.py`:
  - The new code path with `K = Camera.K` and 1920×1200 matches the old one bit for bit.
  - The `site` mode forward/inverse round trip works with the LM and YCB-V cameras.
  - **Done 2026-10-20: 13 tests pass.** The tests needed code, so the `src/utils.py` part of the refactor bullet below was done here:
    - `translation_to_bbox_relative_translation` and `bbox_relative_translation_to_translation` take optional keyword arguments `K` ((3,3) or per-sample (B,3,3)), `img_wh` (tuple or (B,2)), `depth_mode` (`"legacy"` | `"site"`) and `diameter` (scalar or (B,)).
    - With `K is None` and `"legacy"`, the original code runs unchanged, so every existing caller is untouched. New helpers: `_intrinsics`, `_depth_ratio`, `DEPTH_MODES`.
    - `site`: `U_z = t_z·s/(f̄·D)`, with `s = (w+h)/2` and `f̄ = (fx+fy)/2`. The inverse is `t_z = U_z·f̄·D/s`.
  - What the tests show:
    - `K = Camera.K` with `img_wh = (1920, 1200)` matches the legacy path **bit for bit** (`torch.equal`, float32), both directions. Checked with single and per-sample K, and with tuple and tensor `img_wh`.
    - The legacy `U_z = t_z·(w+h)/(2H)` identity holds, and the legacy round trip works.
    - `site` round trip with the LM, YCB-V uw and YCB-V cmu cameras: error < 1e-9 in float64, and < 1e-3 mm in float32.
    - A mixed uw/cmu batch with per-sample K and diameters also round-trips. Using the wrong K for half of it breaks it, so per-sample K really matters.
    - `site` U_z is invariant to (2·D, 2·t) scaling.
    - Argument errors are raised for an unknown `depth_mode`, a missing `img_wh` or a missing `diameter`.
  - **Measured on a synthetic LM batch** (D 100–270 mm, t_z 0.4–1.9 m): `site` U_z = 0.62–1.09, while legacy U_z = 83–321. This confirms §6.2: with legacy, U_z would dominate `RelativeTranslationLoss`.
  - Non-regression: the golden SPEED eval after the change gives 0.0093, and `results.json` is byte-identical (`$NAS_HOME/results/eval-m1-translation-param`). `tests/` now has 35 tests, all passing in 7 s.
- [x] Refactor:
  - Optional `K` / `img_wh` / `depth_mode` / `diameter` in `src/utils.py` (done in the bullet above) and `CoordinateTransformer`.
  - `unpack_batch` in `src/engine.py`.
  - `num_objects` in `FastPoseViT` (default 1).
  - **Done 2026-10-20.**
    - `FastPoseViT(..., num_objects=1)`: the last layer(s) output `num_objects × out_dim`. `forward(x, obj_idx)` returns the sample's own object block (gather), and `forward(x)` with N>1 returns every object as (B, N, D), for export.
      - With N=1 the layers and forward are unchanged, so the published SPEED weights load strictly. Works with the merged/no-MLP and MLP heads.
    - `src/engine.py`:
      - `unpack_batch` takes the SPEED tuple or a BOP dict. The optional keys `K`, `img_wh`, `obj_idx`, `diameter` default to None, and other keys pass through.
      - `forward_model(model, images, obj_idx, merge_outputs)` calls `model(images)` exactly as before when `obj_idx is None`.
      - `ValidationConfig.depth_mode`, defaulting to legacy.
      - `CoordinateTransformer(..., depth_mode)` and `transform_to_absolute(..., K, img_wh, diameter)`. Without K and in legacy mode, the original call is unchanged.
      - `train_epoch` / `val_epoch` use these.
    - Not changed: `evaluate.py`, `predict.py` and `optimization/` still use the SPEED-only path. BOP inference comes with `src/predict_bop.py` (M6).
  - Tests:
    - `tests/test_multi_object_heads.py`, 11 tests: N=1 layout; SPEED weights load; gather for all 4 head types; gradients reach only the sample's own object head; argument check.
    - `tests/test_engine_batch.py`, 5 tests: tuple/dict unpacking; `forward_model` split and `obj_idx`; `CoordinateTransformer` with `K = Camera.K` equal (`torch.equal`) to legacy; `site` mode inverts the dataset transform with LM K.
    - **51 tests pass** (21 s).
  - Non-regression:
    - Golden SPEED eval: 0.0093, `results.json` byte-identical (`results/eval-m1-refactor`).
    - 2-epoch SPEED smoke run in `dev` with the `smoke-02` command (`results/smoke-m1-refactor`): train loss 0.4980 / 0.1701 vs 0.4929 / 0.1710, val score **0.1317** vs 0.1400 (smoke-01 gave 0.1379). Training is unseeded, so this is within noise.
    - Epoch 2 took 6.7 min vs 1.8 min in `smoke-02`, while epoch 1 took 4.3 min in both. The GPU sat at 0 % at times, so it's I/O-bound: the NAS page cache was warm in `smoke-02`'s epoch 2. It isn't the code (the golden eval timing is unchanged). Keep it in mind for M2 throughput.
- [x] Checkpoint/resume in `src/train.py`:
  - Atomic `last.pt` in `$RUN_DIR` containing model, optimizer (Muon), scheduler/LR index, GradScaler, epoch, global iteration, best metric and RNG states.
  - `--resume auto`.
  - Test: kill a 3-epoch smoke run after epoch 1, resume, and check that the curve continues.
  - **Done 2026-10-20.**
    - `src/checkpoint.py`: `save_checkpoint` (atomic: `.tmp` then `os.replace`), `load_checkpoint`, `get_rng_state` / `set_rng_state` (python, numpy, torch, cuda), `unwrap_model` (torch.compile) and `resolve_resume_path`.
    - `src/train.py`:
      - `--checkpoint_path` defaults to `LOG_DIR/last.pt` (= `$RUN_DIR/last.pt` in jobs).
      - `--resume auto` resumes if that file exists and otherwise starts fresh, so a restarted job can re-run the same command. `--resume PATH` requires the file to exist.
      - `last.pt` is saved every epoch, after the best-model check. It holds model (unwrapped), optimizer (Muon / Adam), scheduler, GradScaler, epoch, global iteration (which indexes Muon's precomputed LRs), best val loss, RNG states and args.
      - On resume, TensorBoard is re-opened with `purge_step` and changed args are logged as a warning.
    - Without `--resume` the behaviour is the same as before, plus the `last.pt` write: **~690 MB**, ~1 s per epoch for ViT-B/16-384 + Muon.
  - Unit tests `tests/test_checkpoint.py` (6):
    - A run interrupted after 3 of 6 steps, rebuilt with a different seed, then resumed from `last.pt` gives **bit-identical** weights to the uninterrupted run, for Muon+aux-Adam and for Adam + `CosineAnnealingWithWarmRestarts`. RNG is drawn from python, numpy and torch every step.
    - A control test shows that skipping the RNG restore diverges.
    - Also tested: atomic replace (no `.tmp` left), the compiled-model unwrap, and `resolve_resume_path`.
    - **57 tests pass.**
  - **Kill/resume smoke run** in `dev` (`results/smoke-m1-resume`; driver: 3-epoch `smoke-02` command + `--resume auto`). The process group was **SIGKILLed** right after the epoch-1 checkpoint, then the identical command was re-run:
    - Run 1 logged "No checkpoint … starting from scratch". Epoch 1: train 0.5185, val score 0.2137, LR 2.0000e-4 → 1.9516e-4.
    - Run 2 logged "Resumed … 1 epoch(s) done, global iteration 1200, best validation loss 0.2137".
      - Epoch 2 LR **1.9515e-4** → 1.8111e-4. This is the next step and exactly smoke-02's epoch-2 LR range, so the schedule continues.
      - Train 0.1761, val 0.1403 (smoke-02: 0.1710 / 0.1400).
      - Epoch 3: train 0.1215, val **0.1005** (speed-short-001, 3 epochs uninterrupted: 0.1021).
      - Best model and checkpoint were updated each epoch. Exit 0, no `.tmp` and no stray processes left.
- **Exit:** golden diff = 0 (≤ 1e-6), pytest green, 2-epoch SPEED smoke run behaves as before. **Met 2026-10-20:** golden byte-identical, 57 tests pass, smoke val 0.1317 (2 ep), and the kill/resume run behaves as before.

### M2 — BOP data loader for LM-O (W3–4)
- [x] `src/bop_dataset.py` index: one row per visible instance (`visib_fract ≥ 0.1`, valid bbox), stored as numpy arrays and cached as `.npz` under `$NAS_HOME/cache/`. Expect ≈ 343k train rows and 1,445 LM-O test targets. **Done 2026-10-21.**
  - API: `load_index(dataset, split)` builds the index on first use and caches it in `$NAS_HOME/cache/bop_index/<ds>_<split>_<hash>.npz`. The hash covers the sources, objects, threshold, BOP root and `INDEX_VERSION`, so changed settings never reuse a stale file. The save is atomic. `image_path(index, row)` gives the RGB path. CLI: `PYTHONPATH=. .venv/bin/python -m src.bop_dataset` builds and summarizes all splits.
  - Row fields: `source, scene_id, im_id, gt_id, obj_id, obj_idx` (head index = position in `LMO_OBJ_IDS`), `R` (3×3 `cam_R_m2c`), `t` (mm), `K` (per image), `img_wh`, `bbox_obj`, `bbox_visib` (xywh), `visib_fract`, `px_count_visib`. Per-object tables: `obj_ids`, `diameters` (from `lmo/models_eval`), `symmetric`.
  - Splits in `SPLITS['lmo']`: `train` = LM PBR scenes 0–47; `val_pbr` = PBR 48–49; `val_real` = LM `test` (bop19) scenes 1, 5, 6, 8, 9, 10, 11, 12 via their targets; `val` = both; `test` = LM-O bop19 targets. Target-based splits keep exactly the targets (no visibility filter) and fail loudly if any target is missing from `scene_gt`.
  - Counts:
    | split | rows | images | per object (1, 5, 6, 8, 9, 10, 11, 12) |
    |---|---|---|---|
    | train | **320,959** | 47,987 | 38,238 · 41,561 · 40,133 · 42,214 · 38,673 · 40,108 · 40,140 · 39,892 |
    | val_pbr | 13,484 | 1,998 | 1,582 · 1,748 · 1,722 · 1,788 · 1,597 · 1,709 · 1,686 · 1,652 |
    | val_real | 1,600 | 1,600 | 200 each |
    | test | **1,445** | 200 | 175 · 199 · 171 · 200 · 180 · 180 · 140 · 200 |
  - The ≈ 343k estimate was for all 50 PBR scenes; 48 training scenes give 321k (about 6.7k per scene), well balanced across objects (38k–42k).
  - Building all splits takes about 12 s from the NAS; loading the 77 MB train cache takes 0.2 s.
  - Findings:
    - **LM-O test GT rotations are not exactly orthonormal** (max |RRᵀ − I| = 0.0096, det up to 1.014; PBR and LM test are within 2e-6). Keep this in mind for the label round-trip tolerance and for any test that assumes `R⁻¹ = Rᵀ` on LM-O test.
    - LM-O test targets include instances that are barely visible (`visib_fract` down to 0.003). The index keeps them, since BOP scores every target.
    - `val_real` instances are all well visible (`visib_fract` ≥ 0.66): it is an easy real-image check, not a stand-in for occlusion.
  - Tests: `tests/test_bop_dataset.py` (7) recounts PBR scene 0 from the raw JSON (6,866 rows), checks the test rows equal the 1,445 targets, compares 20 random rows per split field by field with the raw files, checks the object table, the cache round trip and that val excludes the LM-O test scene. **64 tests pass.**
- [x] `__getitem__`: **Done 2026-10-21** (`BOPDataset` in `src/bop_dataset.py`).
  - Crop from `bbox_visib` with jitter, as a single `warpAffine` to a square crop with zero padding.
  - Labels: allocentric R and `U` in `site` mode; also return K, `obj_idx`, diameter and ids.
  - Pixel augs without OpticalDistortion.
  - Crop: `crop_warp_matrix` + `crop_image`, one `cv2.warpAffine` (bilinear) of the square of side max(w, h) centred on the bbox, straight to the model size. Coordinates are pixel indices, the convention of BOP bboxes and K, so the bbox centre maps exactly to the crop centre (S−1)/2. As in SPEED, the bands of the square outside the bbox are black (`crop_context=True` fills them with the image instead: possible ablation). Outside the image is zero.
  - Bbox: `bbox_visib` by default (`bbox_source="obj"` for `bbox_obj`). Training jitter reuses SPEED's `random_bbox_augmentation` (±10 % per side, with probability 0.5, as `SpatialDataAugmenter`). Labels are computed from the jittered bbox, as in SPEED. No rotation augmentation (§6.3).
  - Labels: `U` from `translation_to_bbox_relative_translation` with per-sample K, `depth_mode="site"` (default) and the object diameter; rotation = SPEED's allocentric conversion (`rotation_matrix_to_quaternion` → `get_apparent_orientation` → matrix), so `CoordinateTransformer` inverts it unchanged. Computed in float64, returned as float32.
  - Sample dict (goes straight through `unpack_batch`): `images, translations, rotations, bboxes, K, img_wh, obj_idx, diameter`, plus `R_m2c, t_m2c` (original BOP pose) and `obj_id, scene_id, im_id, gt_id, row` for tracing.
  - Pixel augs: `pixel_aug="standard"` (default: SPEED's standard pipeline without Resize), `"domain_gap"` (SPEED's domain-gap pipeline without Resize and **without OpticalDistortion**) or `"none"`. Only used when augmenting (default: split `train` only).
  - **Finding: `domain_gap` is far too strong for LM** (visually; **overturned by the M3 ablations on 2026-10-22: it gives the best real-image results**). Eleven transforms at p = 0.5 each (incl. solarize, sun flare, tone curve, gray) made most crops in a preview grid unrecognizable. Hence `standard` as default; compare both in the M3 ablations (new decision D5 in §8).
  - Speed: one sample in ~3–30 ms (single process, warm cache); a first 32-sample batch with 8 workers in 2.2 s (cold). Proper throughput measurement is a later M2 bullet.
  - Tests: 9 new in `tests/test_bop_dataset.py`: the warp maps the bbox centre to the crop centre with one scale; black bands vs context; augmentations are photometric only (no OpticalDistortion/Resize); fields, dtypes and an orthonormal label R; eval samples are deterministic; the projected 3D centre lands in the crop exactly where U_x, U_y say (1e-5); `site` and `legacy` U_z match their formulas; jitter stays inside the image and is applied about half of the time; a collated batch passes `unpack_batch`. **73 tests pass.**
- [x] Reprojection test: project `models_eval` points with the GT pose. The projected bbox must match `bbox_obj` with IoU ≥ 0.9, and using `Rᵀ` must clearly fail. Run it on 30 PBR and 30 LM-O instances. **Done 2026-10-21.**
  - `tests/test_bop_dataset.py::test_reprojected_models_match_bbox_obj`, on 30 random instances from PBR scenes 0, 17, 33 and 47, and 30 from LM-O test. Only boxes fully inside the image are sampled, because BOP clips `bbox_obj` at the border.
    | | IoU with R: min / mean | mean edge offset (x1, y1, x2, y2), px | IoU with Rᵀ: mean, how many < 0.9, max |
    |---|---|---|---|
    | PBR | 0.964 / 0.982 | −0.02, 0.03, 0.07, 0.05 | 0.724, 25/30, 0.986 |
    | LM-O test | 0.971 / 0.988 | 0.06, 0.03, 0.03, −0.04 | 0.709, 27/30, 0.988 |
  - Rᵀ fails clearly on average, but a few instances still overlap well: a box is a weak cue for rotations near the identity or near a box-preserving turn. The test therefore asserts mean IoU < 0.8, at least 60 % below 0.9, and a gap > 0.2 to the correct R. The pose-level check is the label round trip (next bullet).
  - **Finding: BOP's pixel convention, fixed in `src/bop_dataset.py`.** The first run had the left/top edges matching to 0.03 px but the right/bottom edges exactly **+1.0 px** off (200 instances each, both datasets). So in BOP, pixel i covers [i, i+1) in the coordinates K projects to (centre at i + 0.5), and `[x, y, w, h]` spans [x, x+w+1] × [y, y+h+1].
    - `bop_bbox_to_xyxy` now returns `[x, y, x+w+1, y+h+1]`, and `crop_warp_matrix` maps those K coordinates to the crop with the half-pixel offsets. The bbox centre lands on the crop centre, and the object's projected centre lands exactly where U says (tested to 1e-5 px).
    - Before the fix, the U labels were shifted by about half a pixel from where the object appears in the crop: ~0.008 in U_x for a 63 px box, about 1 mm laterally. Small but systematic. IoU went from a mean of 0.957 to 0.982 with the fix.
    - Use `bop_bbox_to_xyxy` for every BOP box, including the detections in M7.
  - **75 tests pass.**
- **Visualisations (added 2026-10-21, on request between bullets):** `vis/` has one script per figure and `vis/README.md` explains each one and what to look for; `bash vis/run_all.sh` regenerates all 11 PNGs in `vis/figures/` in ~50 s. Covers M0/M1 (golden SPEED errors, refactor and kill/resume runs) and M2 so far (objects, full scenes with both boxes, model-input crops with GT points and the pose decoded from the labels, augmentations, reprojection and the +1 px convention, index statistics incl. legacy vs site U_z). Observations: LM-O test is more occluded than PBR and in a narrower distance range (~600–1,300 mm vs 300–1,900 mm); site U_z peaks around 0.8 for both. `vis/crops.py --per_object 8` already gives the 64-crop grid of the visual-check bullet below.
- [x] Label round trip: dataset labels → inverse functions (K, bbox) → GT ego pose, within 1e-4. **Done 2026-10-21.**
  - Decoded with `CoordinateTransformer.transform_to_absolute`, the exact path the model's predictions take in validation, on collated **float32** batches as in training (labels in place of predictions).
  - Results (max over the samples; t relative error, R max |ΔR| entry vs the GT rotation):
    | samples | t | R |
    |---|---|---|
    | 256 PBR, jitter on every crop, `site` | 5.3e-7 | 9.7e-7 |
    | same, `legacy` depth | 6.4e-7 | 1.3e-6 |
    | same, `site`, no rotation compensation | 4.8e-7 | 4.0e-7 |
    | all 1,445 LM-O test targets | 2.4e-7 | 7.4e-6 |
    | all 1,600 LM real val | 2.5e-7 | 6.0e-6 |
  - **Finding and fix:** before the fix, LM-O test rotations came back up to **2.9e-3** from the nearest valid rotation (~0.2°). Their GT is slightly non-orthonormal (index bullet), and the quaternion conversion behind the labels projects such a matrix onto *a* rotation, not the nearest one. `nearest_rotation` (SVD) now snaps R before the labels are built; it changes PBR/LM rotations by ~1e-6. The index keeps the raw GT (`R_m2c`), which is what BOP evaluates against.
  - Tests: 5 new (3 training-crop variants, LM-O test, `nearest_rotation`). **80 tests pass.**
- [x] Visual check: a 64-crop grid with projected axes/points saved as PNG, including eggbox and glue. **Done 2026-10-21.**
  - `vis/figures/03_crops_{train,test,val_real}_64.png` (`.venv/bin/python vis/crops.py --split S --per_object 8 --seed 1`): 8 crops of each of the 8 objects per split, train with jitter + colour augmentation. Dots: GT model points warped into the crop. Axes and cross: the pose decoded from the labels with `CoordinateTransformer`'s inverse functions.
  - Inspected all 192 tiles: the points cover the object in every tile, the cross sits on the model origin and the axes follow the object, for eggbox and glue too, in PBR, LM real and LM-O test. No misplaced or mirrored overlay.
  - What the grids show about the data:
    - LM-O test has many heavily occluded targets (e.g. eggbox 8 %, glue 13 % visible): the crop then shows mostly the occluder, and the object centre can lie outside the crop. Expected, and the main difficulty of LM-O.
    - PBR crops have random backgrounds and lighting (e.g. magenta textures); LM real crops are clean and well visible (`val_real` ≥ 66 % visible), so `val_real` measures the synthetic-to-real gap, not occlusion.
    - Eggbox and glue look the same after their 180° symmetry, so their rotation labels are ambiguous in some views: this is what M5's symmetry-aware loss is for.
- [x] Loader throughput with 16 workers. If it is below about 2× GPU speed, stage the PBR RGB (~5.4 GB) to local disk or group crops per image. **Done 2026-10-21: 2.8× the GPU, no staging needed.**
  - Benchmark: `scripts/bench_throughput.py` (`loader`, `sample`, `gpu` modes), in `dev` (A100-80GB, 16 CPUs). Train split, 384 px crops, batch 32, shuffled.
    | what | samples/s |
    |---|---|
    | DataLoader, 16 workers, `standard` pixel aug | **730** (550 including worker start-up) |
    | DataLoader, 8 workers | 553 |
    | DataLoader, 16 workers, no pixel aug | 878 |
    | GPU train step, ViT-B/16-384, 8 heads, AMP + Muon, batch 32, `torch.compile` max-autotune | **257** (125 ms/step; compile takes ~100 s) |
    | same without compile | 225 |
    | same, batch 8 / batch 64, without compile | 99 / 287 |
  - One sample in one process: 5.9 ms read + JPEG decode, 10.4 ms crop + augmentations + labels. A repeated pass over the same (now cached) images gave the same 730/s, so the loader is **CPU-bound, not NAS-bound**. The NAS slowness seen in M1 (SPEED's large PNGs) doesn't apply to the small PBR JPEGs.
  - Batch 8 (the SPEED default) wastes the A100: about 2.6× slower per sample than batch 32. The fixed per-step cost (Muon's orthogonalisation) dominates at small batch. Use batch 32 for BOP and check the LR scaling (§6.7).
  - Implications:
    - A full pass over the 321k training crops takes ~21 min. The 310-epoch warm-restart schedule with 25k-sample virtual epochs (7.75M samples) takes ~8.5 h on one A100, so it fits in one 12 h job.
    - For multi-GPU (M6): 2 GPUs need ~515 samples/s. 16 CPUs give 730/s, only 1.4× that, so request about 16 CPUs per GPU (32 for 2 GPUs) to keep the loader from becoming the limit.
- [ ] (Optional, **deferred 2026-10-21** by the user: revisit if M3/M4 overfit to viewpoints) exact roll augmentation, off by default, tested by checking projected points stay inside the warped mask.
- **Exit:** all tests pass and the overlays look right. **Met 2026-10-21:** 81 tests pass, 64-crop grids look right, loader 2.8× GPU speed. M2 closed (optional roll augmentation deferred).

### M3 — Single-object baseline B1: original architecture, GT boxes (W4–5)
- [x] **Image `fastposevit:1.60` = 1.59 + `libegl1` (moved from M0 to M2, then to M3 on 2026-10-20; done 2026-10-21).** It is not blocking: the EGL workaround (`$NAS_HOME/.local/lib/egl`, used automatically by `scripts/bop_eval.sh`) works in `dev` and `train`.
  - `Dockerfile` already lists `libegl1` (2026-10-20), so any future full rebuild includes it.
  - A thin layer on top of 1.59 keeps the Python environment bit-identical, which a full rebuild with unpinned requirements might not (golden SPEED check).
  - First attempt, 2026-10-20 from home: cancelled. The pull of 1.59 ran at ~70 KB/s.
  - **Built and pushed 2026-10-21** (digest `sha256:8683fbb3…`). The overnight pull took ~2 h 20 min; the first build then failed with a Docker Desktop snapshotter error (`failed call to UtimesNanoAt … snapshots/…/fs/etc: no such file or directory`), and simply re-running `docker build` worked (cached layers, `apt-get` ~62 s). The push uploaded only the new layer. `rcp/project.env` now points to 1.60.
  - **Verified 2026-10-21** (`dev` restarted on 1.60, A100; results in `results/eval-m3-image160/`):
    - `ldconfig -p` lists `/usr/lib/x86_64-linux-gnu/libEGL.so.1`, so `bop_eval.sh` no longer uses the workaround (kept as a fallback for 1.59).
    - BOP GT-as-prediction on LM-O test: AR = **1.0** (VSD, MSSD, MSPD), 50 s in `dev`.
    - Golden SPEED eval: **0.0093**, `results.json` byte-identical to `eval-rcp-check`.
    - `pytest tests/`: 57 passed.
    - Train job `lmo-m3-image160-001` (1.60, 4 CPUs): libEGL found, AR = 1.0, eval 77 s, exit 0; job stopped.
    - The leftover jobs `lmo-m0-access-003` and `lmo-m0-subdebug-001` were deleted by the user.
  - Steps for the user, in a **WSL terminal** with Docker Desktop's WSL integration on. Run them **on the EPFL network**: at home the pull ran at ~70 KB/s, which means more than a day for 1.59's ~7 GB. Run each command separately:
    1. Log in, with the username `sarro` and the **Harbor CLI secret** from registry.rcp.epfl.ch → your name → User Profile (not the EPFL password):
       ```bash
       docker login registry.rcp.epfl.ch -u sarro
       ```
    2. Create a build folder and go into it:
       ```bash
       mkdir -p ~/fastposevit-1.60 && cd ~/fastposevit-1.60
       ```
    3. Write the 3-line Dockerfile (paste all 5 lines at once):
       ```bash
       cat > Dockerfile <<'EOF'
       FROM registry.rcp.epfl.ch/cvlab-pablo/fastposevit:1.59
       USER root
       RUN apt-get update && apt-get install -y --no-install-recommends libegl1 && rm -rf /var/lib/apt/lists/*
       EOF
       ```
    4. Check that it prints exactly the 3 lines `FROM …`, `USER root`, `RUN …`:
       ```bash
       cat Dockerfile
       ```
    5. Build. This pulls 1.59 first, and finished layers are reused if an earlier pull was interrupted:
       ```bash
       docker build -t registry.rcp.epfl.ch/cvlab-pablo/fastposevit:1.60 .
       ```
    6. Push. Most layers say "Layer already exists", and it ends with `1.60: digest: sha256:…`:
       ```bash
       docker push registry.rcp.epfl.ch/cvlab-pablo/fastposevit:1.60
       ```
  - Then Claude sets `IMAGE=...:1.60` in `rcp/project.env`. The user restarts `dev` from the jumphost:
    ```bash
    runai delete job fastpose-dev
    ```
    ```bash
    dev
    ```
  - Then Claude checks that `ldconfig -p | grep libEGL.so.1` finds the library, that `bash scripts/bop_eval.sh` gives AR 1.0 without the workaround, and that the golden SPEED eval still gives 0.0093.
- [x] Overfit 256 instances of one object to near-zero loss in about 10 min. This catches label bugs. **Done 2026-10-21: passed.**
  - **`src/train.py` now trains on BOP** (additive; the SPEED path is unchanged):
    - `--dataset LMO` builds `BOPDataset` train/val from the cached index, with one head per LM-O object (`num_objects=8`).
    - `--bop_obj_ids 8` restricts both splits to some objects (heads keep their LM-O positions). `--bop_val_split val|val_pbr|val_real` (never test). `--max_val_samples N` for a fixed random val subset.
    - `--depth_mode` (default `site` for BOP, `legacy` for SPEED), `--bop_pixel_aug standard|domain_gap`, `--crop_context`. The existing `--no_pixel_augmentation`, `--no_spatial_augmentation`/`--no_bbox_augmentation` and `--bbox_crop_percent` map onto BOP's pixel augmentation and jitter.
    - `--bop_overfit N [--bop_overfit_seed S]`: N fixed training instances, no augmentation, validated on the same instances.
    - Helpers `select_rows`/`rows_for_objects` in `src/bop_dataset.py` (+1 test, **81 pass**). `BOPDataset.camera` exists because `val_epoch` reads it (samples carry their own K).
  - Run `results/lmo-m3-overfit` (in `dev`, 633 s incl. ~100 s compile): driller (8), 256 instances, README best SPEED config (6D relative Frobenius + relative translation, Muon, max_lr 1e-4 → 8e-4 after the batch scaling, warm restarts T0 = 10, Tmult = 2) at batch 32, 150 epochs = 1,200 steps.
    | epoch | train loss | val translation error | val rotation error |
    |---|---|---|---|
    | 1 | 2.93 | 856 mm (rel 0.81) | 98° |
    | 10 (end of cycle 1) | 0.13 | 33 mm | 3.3° |
    | 70 (end of cycle 3) | 0.004 | 1.1 mm | 0.04° |
    | **150 (end)** | **0.0015** | **0.43 mm (rel 4e-4)** | **0.017°** |
  - Each warm restart (epochs 10, 30, 70) briefly raises the error, and each cycle ends lower than the one before: the labels are learnable and consistent, and crop → labels → model → inverse transforms → metrics work end to end. Figure: `vis/figures/10_lmo_m3_overfit.png` (`vis/training_curves.py RUN_DIR ...` plots any run).
- [x] Train FastPose-ViT (ViT-B/16-384, best SPEED config except pixel aug on, `site` depth) on **one object** (8 driller or 5 can) from LM PBR. **Done 2026-10-21:** job `lmo-m3-b1-001` (driller, 150 epochs, ~7.5 h), best val loss 0.0308, final val ‖Δt‖/‖t‖ ≈ 1.0 %, rotation ≈ 1.7°; curves in `vis/figures/10_training_curves.png`; job deleted.
- [x] Evaluate on the LM real val split and on LM-O test for that object: rotation error (°), translation error (mm), ADD@0.1d, MSSD/MSPD. **Done 2026-10-21** with `src/bop_evaluate.py` (GT visib boxes, no jitter, run in `dev`, ~1 min); results in §9 and `$RUN_DIR/eval/` (`summary.json`, per-instance `errors_*.npz`, BOP CSVs). VSD / BOP19 AR is not computed: it needs all 8 objects in the CSV (M5). Translation, not rotation, is the weak part (see §9).
- [x] Ablations at **384 px, 150 epochs (B1's setting), 5 parallel jobs** (decided 2026-10-21, instead of short 224 px / 30-epoch runs): legacy vs `site` depth; relative vs decoupled translation loss; `standard` vs `domain_gap` pixel aug (D5); plus two exact repeats of B1 for the run-to-run noise.
  - `scripts/m3_ablations.sh VARIANT` (one variant per job; `VIT`/`EPOCHS` env vars, default `vit_b_16_384`/150; 224 px / 30 epochs via `VIT=vit_b_16 EPOCHS=30`): trains into `$RUN_DIR/<variant>/`, then evaluates `last.pt` with `src/bop_evaluate.py` on LM real val and LM-O test (`<variant>/eval/summary.json`), then writes `DONE`. B1's `model.pth` was saved at its final epoch, so its evaluation equals a `last.pt` evaluation: directly comparable.
  - Dry run in `dev` (`results/m3-ablation-smoke384`, `dec`, 1 epoch at 384): train → eval → `DONE` in 4 min.
  - The earlier 224 px chain jobs `lmo-m3-abl-001`/`-002` were stopped before finishing (superseded by this decision); their folders are kept but not used.
  - **Launched 2026-10-21 22:23, all Running with data loaded at 22:37** (they were Pending ~10 min waiting for GPUs, then pulled the image):
    | job | variant | differs from B1 by |
    |---|---|---|
    | `lmo-m3-abl-base-001` | base | nothing (repeat) |
    | `lmo-m3-abl-base2-001` | base2 | nothing (repeat) |
    | `lmo-m3-abl-legacy-001` | legacy | `--depth_mode legacy` |
    | `lmo-m3-abl-dec-001` | dec | `--translation_loss decoupled --decoupled_lambda 1.0` |
    | `lmo-m3-abl-domgap-001` | domgap | `--bop_pixel_aug domain_gap` |
  - Each: 1 GPU, 16 CPUs, 32 GiB (as B1). Expected ~7.5–9 h (five jobs share the NAS; domgap has heavier CPU augmentation), so done around 06:00–08:00 on 2026-10-22.
  - **Finished 2026-10-22** (all exit 0, `DONE` written; ends 06:02–06:12, domgap 07:18 = 8.7 h; jobs deleted). Curves: `vis/figures/11_lmo_m3_ablations.png`. Means over n = 200 driller instances per split, GT boxes, `last.pt` (= final epoch).
    | run | LM real val: trans mean / median (mm) | rot (°) | ADD@0.1d | LM-O test: trans mean / median (mm) | rot (°) | ADD@0.1d | ADD@0.05d | recall MSSD / MSPD |
    |---|---|---|---|---|---|---|---|---|
    | B1 | 13.1 / 10.4 | 1.72 | 0.870 | 19.0 / 15.5 | 1.61 | 0.765 | 0.415 | 0.871 / 0.927 |
    | base | 12.9 / 10.4 | 1.55 | 0.875 | 18.3 / 13.9 | 1.57 | 0.760 | 0.460 | 0.878 / 0.935 |
    | base2 | 12.7 / 10.4 | 1.65 | 0.895 | 18.7 / 13.7 | 1.59 | 0.770 | 0.470 | 0.880 / 0.932 |
    | legacy | 15.5 / 12.5 | 1.92 | 0.865 | 20.9 / 17.0 | 1.62 | 0.715 | 0.375 | 0.858 / 0.896 |
    | dec | 13.7 / 10.4 | 1.68 | 0.860 | 18.0 / 13.6 | 1.64 | 0.760 | 0.470 | 0.882 / 0.933 |
    | **domgap** | **10.4 / 8.8** | 1.63 | **0.975** | **17.5** / 13.7 | 1.66 | **0.815** | 0.460 | **0.888** / 0.931 |
  - Run-to-run noise (B1, base, base2): LM-O test trans mean 18.3–19.0 mm, ADD@0.1d 0.760–0.770, MSSD 0.871–0.880; LM real val trans 12.7–13.1 mm, ADD@0.1d 0.870–0.895. Rough statistical scale: with n = 200, the standard error of an ADD@0.1d around 0.77 is ~0.03.
  - Conclusions:
    - **`site` depth confirmed**: `legacy` is worse than every baseline run on both splits (test trans +2 mm, ADD@0.1d −0.05, MSPD −0.03) and needs ~40 epochs before its translation error drops (U_z ~100× larger than U_x, U_y; see `vis/figures/07_index_distributions.png`).
    - **Decoupled translation loss: no measurable effect**: every metric is inside the baseline spread (test trans 18.0 mm is 0.3 mm below it, ADD@0.1d equal). Keep the relative loss (fewer knobs).
    - **`domain_gap` pixel aug helps on real images**, although it looked too strong in the preview (D5): LM real val ADD@0.1d 0.975 vs ≤ 0.895 and trans 10.4 vs ≥ 12.7 mm; LM-O test ADD@0.1d 0.815 vs ≤ 0.770 (~1.5–2 standard errors), trans 17.5 vs ≥ 18.3 mm, MSSD 0.888 vs ≤ 0.880. Rotation is slightly worse (1.66° vs ≤ 1.61°), and its training/val loss is higher, because the mixed val split is half synthetic and the augmentation makes training harder: the gain is on **real** images, i.e. the synthetic-to-real gap. It costs ~15 % more time (CPU-bound augmentation: 8.7 h vs 7.5 h).
    - Translation stays the weak part in every variant (test median ~14 mm, < 5 mm for only 5–11 %).
  - Suggested default from M4 on (to confirm with the user): `site` depth, relative translation loss, `--bop_pixel_aug domain_gap`.
- **Exit:** B1 logged in §9. This is "the existing architecture on BOP".

### M4 — Shared multi-object model (W6–7)
- [x] Per-object heads (option a) with object-balanced sampling, trained on all 8 LM-O objects. **Started 2026-10-22; done 2026-10-25** (6 jobs finished 2026-10-22/23, results below).
  - The heads were already in place since M1/M3 (`num_objects=8`, `FastPoseViT._select_object`).
  - The per-object pools are already almost balanced (LM PBR train: 38,238–42,214 rows per object), so the balanced sampler mainly sets the **epoch size**. A full pass over all 8 objects is 321k crops (~60 h for 150 epochs at 384 px).
  - New `ObjectBalancedSampler` (`src/bop_dataset.py`) and `train.py --bop_epoch_samples N [--bop_sampler_seed S]`:
    - Each virtual epoch draws N/8 crops of every object, then shuffles them.
    - Each object walks through its own permutations, so no crop repeats before its object's pool is used up.
    - The order depends only on (seed, epoch), so a resumed job continues with the same samples.
    - Default `0` = the old behaviour (one shuffled pass).
  - Tests: +2 sampler tests, **88 pass**. Smoke in `dev` (`results/m4-balanced-smoke`, 224 px, 1,280 samples/epoch): 160 × 8 per epoch, val loss decreasing, resume after 2 epochs continues at iteration 80.
  - `scripts/m4_multi.sh VARIANT` (one variant per job): trains all 8 objects into `$RUN_DIR/<variant>/`, then evaluates `last.pt` **per object on LM real val only** (`eval/obj_<id>/summary.json`; LM-O test only at the end of the milestone), then writes `DONE`. Dry run in `dev` (`results/m4-multi-dryrun`, `SPE=640 EPOCHS=1 VIT=vit_b_16`): train → 8 evals → `DONE` in 2 min. This dry run found the next bullet.
  - All jobs: 384 px, 150 epochs, `site` depth, relative translation loss (both settled in M3), batch 32, 1 GPU / 16 CPUs / 32 GiB. 1× = B1's compute (42,208 samples/epoch = 5,276 per object), so at 1× each object gets 1/8 of B1's driller samples. **Launched 2026-10-22 12:10** (Pending at first):
    | job | variant | pixel aug | samples/epoch | sampler seed | expected |
    |---|---|---|---|---|---|
    | `lmo-m4-std-001` | std | standard | 1× | 0 | ~7.5 h |
    | `lmo-m4-std2-001` | std2 | standard | 1× | 1 (repeat) | ~7.5 h |
    | `lmo-m4-dg-001` | dg | domain_gap | 1× | 0 | ~8.7 h |
    | `lmo-m4-dg2-001` | dg2 | domain_gap | 1× | 1 (repeat) | ~8.7 h |
    | `lmo-m4-dg2x-001` | dg2x | domain_gap | 2× | 0 | ~17 h |
    | `lmo-m4-dg4x-001` | dg4x | domain_gap | 4× | 0 | ~35 h |
  - The questions they answer: `standard` vs `domain_gap` aug in the shared model (with 2 seeds each for noise), and whether the shared model needs more than B1's compute (1× vs 2× vs 4×) to reach ≥ 90 % of B1 on the driller (B1 LM real val ADD@0.1d 0.87; the M3 runs give 0.87–0.975).
  - No "unbalanced" variant: the pools are already within 10 % of each other, so plain uniform sampling would be almost the same.
  - **Finished** (all exit 0, `DONE` written): the 1× runs 19:52–20:04 on 2026-10-22 (~7.7 h, std and dg alike), `dg2x` 04:03 on 2026-10-23 (~15.9 h), `dg4x` 21:57 on 2026-10-23 (~33.8 h). Jobs deleted 2026-10-25. Curves: `vis/figures/12_lmo_m4_multi.png`. Full per-object tables (also trans/rot/MSPD): `$NAS_HOME/outputs/FastPose-ViT/runs/m4_val_real_table.txt`.
  - **LM real val, `last.pt`, GT boxes, n = 200 per object** (10 eggbox, 11 glue: ADD-S and symmetric MSSD; the loss is still symmetry-unaware):
    | run | ADD@0.1d: 1 ape | 5 can | 6 cat | **8 driller** | 9 duck | 10 eggbox | 11 glue | 12 holepuncher | mean | MSSD recall mean | trans mean (mm) | rot mean (°) |
    |---|---|---|---|---|---|---|---|---|---|---|---|---|
    | std | 0.440 | 0.580 | 0.545 | 0.785 | 0.370 | 0.805 | 0.565 | 0.380 | 0.559 | 0.744 | 21.4 | 3.07 |
    | std2 | 0.430 | 0.585 | 0.590 | 0.800 | 0.395 | 0.830 | 0.495 | 0.470 | 0.574 | 0.752 | 20.8 | 3.14 |
    | dg | 0.490 | 0.790 | 0.705 | 0.875 | 0.305 | 0.840 | 0.680 | 0.360 | 0.631 | 0.777 | 18.4 | 2.90 |
    | dg2 | 0.545 | 0.755 | 0.680 | 0.845 | 0.425 | 0.850 | 0.595 | 0.345 | 0.630 | 0.775 | 18.8 | 2.93 |
    | dg2x | 0.485 | 0.850 | 0.670 | **0.925** | 0.315 | 0.920 | 0.700 | 0.375 | 0.655 | 0.795 | 17.3 | 2.59 |
    | dg4x | 0.555 | 0.870 | 0.690 | 0.905 | 0.290 | 0.935 | 0.670 | 0.390 | **0.663** | **0.802** | **16.9** | **2.34** |
    | single-object driller (M3): B1 / base / base2 / domgap | | | | 0.870 / 0.875 / 0.895 / 0.975 | | | | | | | 13.1 / 12.9 / 12.7 / 10.4 (driller) | 1.72 / 1.55 / 1.65 / 1.63 (driller) |
  - Run-to-run noise (2 sampler seeds, same flags): the 8-object mean ADD@0.1d differs by 0.015 (std) and 0.001 (dg), MSSD by ≤ 0.008, trans by ≤ 0.6 mm. Single objects vary much more (up to 0.12, e.g. duck dg 0.305 vs dg2 0.425; standard error ~0.035 at n = 200), so **compare variants on the 8-object mean**, not on single objects.
  - Conclusions:
    - **`domain_gap` aug helps again**, now on every summary metric: mean ADD@0.1d 0.63 vs 0.56–0.57, MSSD 0.776 vs 0.748, trans 18.6 vs 21.1 mm (≫ noise). It also gains most on the objects with real-image trouble (can, cat, glue +0.1–0.2). Keep it as the default.
    - **More compute helps, with diminishing returns**: mean ADD@0.1d 0.630 (1×) → 0.655 (2×) → 0.663 (4×); rotation 2.92° → 2.59° → 2.34°, trans 18.6 → 17.3 → 16.9 mm. The 1×→2× step is clearly above noise; 2×→4× adds +0.008 ADD (≈ noise) but rotation and translation keep improving. The training and val curves are still falling at epoch 150 in every run: the shared model is under-trained at B1's budget, as expected when each object gets 1/8 of B1's samples at 1×.
    - **Exit criterion (≥ 90 % of B1 on B1's object, no object at 0 ADD) is met on LM real val**: driller ADD@0.1d is 0.785–0.925 against 0.783 = 90 % of B1's 0.870; against the best single-object config (M3 domgap 0.975 → 0.878) only 2× and 4× pass (0.925, 0.905), and 1× `dg` just misses (0.875 / 0.845). The lowest object is duck (9) with 0.29–0.43, then holepuncher (12) and ape (1): the small objects, where 0.1·d is a tight threshold (ape 102 mm, duck 105 mm diameter → ~10 mm). It will be confirmed on LM-O test at the end of the milestone (next bullet).
    - The shared model is still behind the single-object model on the driller in translation (12.3–14.4 vs 10.4–13.1 mm) and rotation (1.9–2.2° vs 1.6–1.7°).
    - Glue (11) has the largest rotation error (3.7–5.7°) and the lowest MSSD (0.52–0.67): it is symmetric, and the loss is not yet (M5).
  - **M4 config from here on (confirmed by the user 2026-10-25):** `--bop_pixel_aug domain_gap` at **2× compute** (`--bop_epoch_samples 84416`, ~16 h) for the ablations, and **4×** (`168832`, ~34 h) for final models.
- [x] **(Added 2026-10-22, found by the M4 dry run.)** `src/bop_evaluate.py` refused the symmetric objects (eggbox 10, glue 11): `NotImplementedError`. **Done 2026-10-22:** ADD-S (`pose_error.adi`) for symmetric objects; MSSD/MSPD use the BOP19 symmetry set (`misc.get_symmetry_transformations(models_info, max_sym_disc_step=0.01)`); the rotation error is the smallest angle to any symmetric GT pose; `trans_mm` stays plain. The training **loss** is still symmetry-unaware (M5). New `tests/test_bop_evaluate.py` (3 tests):
  - GT·S_k for eggbox and glue gives MSSD = MSPD = 0, rotation < 0.1° (GT and S_k are orthonormal to ~1e-4 only), and ADD-S ≈ 2 mm (nearest model vertex; < 0.02·d).
  - The driller still uses plain ADD.
- [x] Per-object table (ADD(-S), MSSD/MSPD recall) on val and, at the end of the milestone only, on LM-O test. **Val part done 2026-10-25; LM-O test done 2026-10-28** with the final model `tok4x` (`last.pt`, confirmed by the user), run once in `dev` (~2 min): `bop_evaluate.py --splits test --object_conditioning token --out_dir $NAS_HOME/outputs/FastPose-ViT/runs/lmo-m4-tok4x-001/tok4x/eval_test` (own folder, so the val `eval/` is kept). LM-O test, GT boxes, all 1,445 instances:
  | object | n | ADD(-S)@0.1d | @0.05d | MSSD | MSPD | trans mean / median (mm) | rot mean (°) | rot < 5° | val ADD (same model) |
  |---|---|---|---|---|---|---|---|---|---|
  | ape 1 | 175 | 0.474 | 0.126 | 0.706 | 0.929 | 15.1 / 10.4 | 6.20 | 0.674 | 0.620 |
  | can 5 | 199 | 0.784 | 0.477 | 0.862 | 0.940 | 13.9 / 10.4 | 3.35 | 0.829 | 0.890 |
  | cat 6 | 171 | 0.339 | 0.164 | 0.571 | 0.912 | 36.2 / 23.7 | 10.04 | 0.456 | 0.675 |
  | driller 8 | 200 | **0.805** | 0.450 | 0.871 | 0.913 | 18.0 / 13.4 | 3.04 | 0.890 | 0.940 |
  | duck 9 | 180 | 0.189 | 0.044 | 0.579 | 0.900 | 19.7 / 18.2 | 10.84 | 0.250 | 0.315 |
  | eggbox 10 (sym.) | 180 | 0.428 | 0.128 | 0.321 | 0.744 | 129.3 / 50.2 | 18.94 | 0.189 | 0.935 |
  | glue 11 (sym.) | 140 | 0.700 | 0.400 | 0.654 | 0.874 | 58.1 / 25.5 | 7.27 | 0.479 | 0.665 |
  | holepuncher 12 | 200 | 0.370 | 0.100 | 0.662 | 0.913 | 23.5 / 19.0 | 4.21 | 0.700 | 0.335 |
  | **mean over objects** | 1445 | **0.511** | 0.236 | **0.653** | **0.891** | 39.2 / 21.4 | 7.99 | 0.558 | 0.672 |
  | mean over instances | 1445 | 0.513 | 0.237 | 0.660 | 0.892 | 38.0 / 21.0 | 7.82 | 0.571 | |
  - Val → test drops from 0.672 to 0.511 ADD (MSSD 0.809 → 0.653). LM-O test is LM scene 2 with heavy occlusion; LM real val has the objects unoccluded. The drop is mostly occlusion, as for B1 (M3: the val→test gap was mostly translation).
  - **Eggbox collapses** (val 0.935 → 0.428 ADD-S, MSSD 0.32, mean translation 129 mm with median 50 mm, so a tail of very wrong poses): it is the most occluded object in LM-O. Cat and duck also lose a lot (rotation errors ~10°). Driller, can and glue hold up best.
  - Not comparable with §2.2 yet: those are BOP AR (with VSD, real detections); ours here are recalls with GT boxes. The first comparable number comes in M6 (official AR, GT boxes) and M7 (detections).
  - Results: `.../lmo-m4-tok4x-001/tok4x/eval_test/` (`table_test.md`, `summary.json`, `obj_<id>/`, and `fpvit_lmo-test.csv`, the input for the official eval in M6).
  - `src/bop_evaluate.py` now evaluates several objects in one run: `--obj_ids` (default all 8; `--obj_id 8` still works), one model load and one prediction pass per split, then errors object by object.
    - Per-object outputs go to `OUT_DIR/obj_<id>/` (`summary.json`, `errors_<split>.npz`), the layout M4 already used.
    - One BOP CSV per split with every evaluated object (`fpvit_lmo-<split>.csv`): on LM-O test this is the file `scripts/bop_eval.sh` needs for the official AR with VSD (M6).
    - `table_<split>.md` columns: n, ADD(-S)@0.1d, @0.05d, MSSD and MSPD recall, trans mean/median, rot mean, rot < 5°. Symmetric objects are marked "(sym.)". Two summary rows: the mean over objects (LM literature) and the mean over instances (weights as BOP19; equal on LM real val, n = 200 each). `summary.json` holds the same.
    - `--table_only` rebuilds the tables from existing `obj_<id>/` folders, without a model.
    - `LMO_SYMMETRIC_IDS = (10, 11)`, checked by a test against the index's `symmetric` table.
  - Tests: +2 (`test_per_object_table_means`, `test_symmetric_id_table_matches_models_info`), **90 pass**.
  - Regression check (`results/m4-eval-table-check`, `dg4x` `last.pt`, LM real val, ~1.5 min in `dev`): every recall and ADD rate is identical to the job's per-object evaluation. The mm means differ by ≤ 0.01 mm (fp16 autocast with batches that now mix objects).
  - `scripts/m4_multi.sh` now makes one evaluator call (all 8 objects). The tables of the 6 M4 runs were rebuilt with `--table_only`: `$NAS_HOME/outputs/FastPose-ViT/runs/lmo-m4-<variant>-001/<variant>/eval/table_val_real.md`.
  - Val table of the best run so far (`dg4x`, LM real val, GT boxes):
    | object | n | ADD(-S)@0.1d | ADD(-S)@0.05d | MSSD recall | MSPD recall | trans mean (mm) | trans median (mm) | rot mean (°) | rot < 5° |
    |---|---|---|---|---|---|---|---|---|---|
    | 1 ape | 200 | 0.555 | 0.240 | 0.817 | 0.997 | 10.6 | 9.0 | 2.33 | 0.970 |
    | 5 can | 200 | 0.870 | 0.450 | 0.912 | 0.986 | 12.3 | 11.3 | 1.37 | 0.995 |
    | 6 cat | 200 | 0.690 | 0.325 | 0.866 | 0.994 | 12.4 | 12.5 | 2.17 | 0.975 |
    | 8 driller | 200 | 0.905 | 0.640 | 0.936 | 0.975 | 12.3 | 9.6 | 1.91 | 0.985 |
    | 9 duck | 200 | 0.290 | 0.125 | 0.707 | 0.996 | 17.3 | 16.9 | 2.22 | 0.980 |
    | 10 eggbox (sym.) | 200 | 0.935 | 0.555 | 0.799 | 0.994 | 18.2 | 15.1 | 2.14 | 0.970 |
    | 11 glue (sym.) | 200 | 0.670 | 0.310 | 0.657 | 0.958 | 31.3 | 27.0 | 3.69 | 0.840 |
    | 12 holepuncher | 200 | 0.390 | 0.145 | 0.721 | 0.980 | 20.9 | 17.8 | 2.92 | 0.905 |
    | **mean** | 1600 | **0.663** | 0.349 | **0.802** | 0.985 | 16.9 | 14.9 | 2.34 | 0.953 |
  - What the table adds: rotation is nearly solved (≥ 95 % under 5° except glue and holepuncher), and MSPD (the 2D metric) is ≥ 0.96 everywhere. The misses are **depth**: the small objects fail ADD@0.1d with ~10–17 mm translation errors (ape and duck: 0.1·d ≈ 10 mm), and glue has the largest translation error (31 mm) and rotation error.
- [x] (Ablation) per-object class tokens (option b), and a combination of a and b. **Started 2026-10-25; done 2026-10-26** (results below).
  - `FastPoseViT(object_conditioning=heads|token|both)`, `train.py`/`bop_evaluate.py --object_conditioning` (default `heads` = the M4 model so far; constant `OBJECT_CONDITIONINGS` in `src/models.py`):
    - `token` (option b): one shared 9-output head, and a per-object class token = the shared (pretrained) class token + a learned per-object embedding `object_token` (8, 1, 768), added at the ViT input (`VisionTransformer.forward(x, class_token_offset)`).
    - The embeddings start at **zero**, so the model starts exactly as the pretrained one. The sequence length is unchanged (no extra latency), but `obj_idx` becomes a required model input (ONNX: one more input, M9).
    - `both` (a + b): per-object heads (72 outputs) and object tokens.
    - Optimizer: like `class_token`, the embedding is 3D, so it goes to Muon. Newton–Schulz runs per leading index, so each object's token gets exactly the update the class token gets. A test checks this; no optimizer change.
    - `num_objects=1` (SPEED) ignores the conditioning: no token, one head.
  - Tests: +7 in `tests/test_multi_object_heads.py`, **98 pass**:
    - `heads` keeps the M4 state-dict layout (M4 checkpoints load).
    - A `token` model at init equals the shared-head model.
    - A non-zero token changes only its own object's output.
    - `both` shapes; `obj_idx` required; SPEED unaffected; Muon per-token update.
  - Golden SPEED eval after the change (`results/eval-m4-tokens-golden`): **0.0093**, `results.json` byte-identical.
  - Dry run in `dev` (`results/m4-tokens-dryrun`, `tok2x` and `both2x` at `SPE=640 EPOCHS=1 VIT=vit_b_16`, with `torch.compile`): train → 8-object eval → `DONE`. `object_token` is saved in `last.pt`, and each token moved ~0.019 in norm in 40 steps (≈ lr 8e-4 per step, like the class token).
  - New variants in `scripts/m4_multi.sh` (`COND` map → `--object_conditioning` for train and eval). All at the confirmed config (`domain_gap`, **2×** = 84,416 samples/epoch, 384 px, 150 epochs; ~16 h each). With the finished `dg2x` this gives 2 seeds for each of a, b and a + b. **Launched 2026-10-25 19:45** (Pending at first):
    | job | variant | conditioning | sampler seed |
    |---|---|---|---|
    | (finished) `lmo-m4-dg2x-001` | dg2x | heads (a) | 0 |
    | `lmo-m4-dg2x2-001` | dg2x2 | heads (a) | 1 (repeat) |
    | `lmo-m4-tok2x-001` | tok2x | token (b) | 0 |
    | `lmo-m4-tok2x2-001` | tok2x2 | token (b) | 1 |
    | `lmo-m4-both2x-001` | both2x | both (a + b) | 0 |
    | `lmo-m4-both2x2-001` | both2x2 | both (a + b) | 1 |
  - Decide on the 8-object mean of LM real val (ADD(-S)@0.1d, MSSD), with the two seeds of each as the noise. The 1× seed pairs differed by ≤ 0.015 on the mean ADD.
  - **Finished 2026-10-26** 10:58–11:38 (all exit 0, `DONE`; ~15.3–15.9 h). Jobs deleted 2026-10-26. Curves: `vis/figures/13_lmo_m4_conditioning.png` (all six almost on top of each other). Table: `$NAS_HOME/outputs/FastPose-ViT/runs/m4_conditioning_table.txt`.
    - `tok2x`: the first attempt crashed 2.5 min in, during `torch.compile` (`OSError: [Errno 116] Stale file handle` on the NAS). runai restarted it at once, from scratch (no epoch done), and it finished normally. runai still shows the job as `Error` (status of the first attempt); `log.txt` is the reliable signal.
    - `tok2x2` was preempted once and resumed at epoch 136 (`--resume auto`, same sampler order).
  - **LM real val, GT boxes, n = 200 per object, `last.pt`** (ADD-S for 10, 11):
    | conditioning | run | 1 | 5 | 6 | 8 | 9 | 10 | 11 | 12 | **mean ADD(-S)@0.1d** | MSSD | MSPD | trans (mm) | rot (°) |
    |---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
    | heads (a) | dg2x | 0.485 | 0.850 | 0.670 | 0.925 | 0.315 | 0.920 | 0.700 | 0.375 | 0.655 | 0.795 | 0.983 | 17.3 | 2.59 |
    | heads (a) | dg2x2 | 0.545 | 0.780 | 0.755 | 0.920 | 0.335 | 0.905 | 0.700 | 0.375 | 0.664 | 0.796 | 0.984 | 17.2 | 2.65 |
    | token (b) | tok2x | 0.540 | 0.785 | 0.770 | 0.905 | 0.335 | 0.955 | 0.770 | 0.375 | **0.679** | **0.806** | 0.984 | **16.4** | 2.62 |
    | token (b) | tok2x2 | 0.580 | 0.785 | 0.715 | 0.935 | 0.330 | 0.920 | 0.700 | 0.395 | 0.670 | 0.799 | 0.983 | 16.9 | 2.67 |
    | both (a + b) | both2x | 0.525 | 0.815 | 0.675 | 0.910 | 0.400 | 0.925 | 0.730 | 0.300 | 0.660 | 0.794 | 0.984 | 17.3 | 2.58 |
    | both (a + b) | both2x2 | 0.525 | 0.870 | 0.705 | 0.885 | 0.320 | 0.910 | 0.705 | 0.365 | 0.661 | 0.797 | 0.983 | 17.1 | 2.58 |
    | heads, 4× (reference) | dg4x | 0.555 | 0.870 | 0.690 | 0.905 | 0.290 | 0.935 | 0.670 | 0.390 | 0.663 | 0.802 | 0.985 | 16.9 | **2.34** |
  - Seed means: heads 0.660, **token 0.675**, both 0.661 (mean ADD); MSSD 0.796 / 0.803 / 0.796; trans 17.3 / 16.7 / 17.2 mm; rot 2.62 / 2.65 / 2.58°. The seed pairs differ by ≤ 0.009.
  - Conclusions:
    - **The three options are close.** The training curves overlap, and every option clears the exit criterion on val (driller 0.885–0.935).
    - **Per-object class tokens (b) are slightly best**: both token runs (0.670, 0.679) beat both heads runs (0.655, 0.664) on the mean ADD, by +0.015 on average, about 2× the seed spread. Translation improves by 0.6 mm and MSSD by 0.007. Rotation shows no gain. With 2 seeds this is suggestive rather than conclusive.
    - **a + b gives nothing over a alone** (0.661 vs 0.660): the token does not add to the heads.
    - Token at 2× is above heads at 4× on the mean ADD (0.675 vs 0.663), but not on rotation (2.65° vs 2.34°): more compute mainly helps rotation.
    - Practical trade-off: `token` has one shared 9-output head (smaller) but needs `obj_idx` as a model input in ONNX/TensorRT (M9); `heads` can output all objects and select afterwards, so it needs no extra input.
  - **Final M4 model (confirmed by the user 2026-10-26): `token` at 4×, 2 seeds.** Variants `tok4x`/`tok4x2` in `scripts/m4_multi.sh` (`--object_conditioning token`, `domain_gap`, 168,832 samples/epoch, seeds 0/1). **Launched 2026-10-26 ~15:10**, both Running at once: `lmo-m4-tok4x-001`, `lmo-m4-tok4x2-001` (~34 h, so done ~01:00 on 2026-10-28). Then the LM-O test table of the final model (next bullet's test part) and the M4 exit check.
    - **Result (2026-10-28; `tok4x` exit 0 at 03:56, `tok4x2` at 07:04, both deleted).** The cluster preempted them ~25 times on 2026-10-27 (no crashes; each resumed from `last.pt`), so they ended ~6 h late. LM real val, GT boxes, `last.pt`, n = 200 per object, mean over the 8 objects:
      | run | ADD(-S)@0.1d | @0.05d | MSSD | MSPD | (MSSD+MSPD)/2 | trans mean (mm) | rot mean (°) |
      |---|---|---|---|---|---|---|---|
      | `dg4x` (heads, 4×) | 0.663 | 0.349 | 0.802 | 0.985 | 0.894 | 16.9 | 2.34 |
      | **`tok4x`** (token, 4×, seed 0) | **0.672** | **0.370** | **0.809** | 0.985 | **0.897** | **16.4** | 2.36 |
      | `tok4x2` (token, 4×, seed 1) | 0.654 | 0.354 | 0.797 | 0.985 | 0.891 | 17.1 | 2.39 |
      | `tok2x` / `tok2x2` (token, 2×) | 0.679 / 0.670 | 0.344 / 0.333 | 0.806 / 0.799 | 0.984 / 0.983 | 0.895 / 0.891 | 16.4 / 16.9 | 2.62 / 2.67 |
      - Token at 4× (mean of 2 seeds 0.663 ADD, 0.803 MSSD) = heads at 4× (0.663, 0.802) = token at 2× (0.675, 0.803): all within the seed spread (0.018 ADD between `tok4x` and `tok4x2`). 4× mainly lowers the rotation error (2.4° vs 2.6°) and helps the driller (0.94–0.95 vs 0.905–0.935); the 2× → 4× compute buys nothing on the mean.
      - Per object (`tok4x`): driller 0.940 (B1 0.870), lowest duck 0.315; glue is the weakest on MSSD (0.656) with ~32 mm translation error.
      - **Proposed final model: `tok4x` `last.pt`**, the best run on the BOP score (MSSD+MSPD)/2 and on ADD@0.1d/0.05d among the 4× runs, chosen on val only. The difference to the others is within noise, so this is a pick, not a finding.
- **Exit:** one network reaches ≥ 90 % of B1 on B1's object, and no object is at 0 ADD. **Met 2026-10-28 on LM-O test:** `tok4x` driller ADD@0.1d 0.805 = 105 % of B1's 0.765 (M3 `domain_gap` single-object 0.815), lowest object duck 0.189 (> 0). **M4 closed.**

### M5 — Symmetries (W7–8)
- [x] `SymmetricRotation6DLoss` with a padded symmetry table. Unit tests:
  - Loss = 0 for every `R_gt·S_k`.
  - Equal to the plain loss for non-symmetric objects.
  - Finite gradients.
  - **Done 2026-10-26** (started early, while the M4 final-model jobs train; it does not depend on them):
    - `symmetry_table(bop_root, dataset, obj_ids)` (`src/bop_dataset.py`): (n_obj, K, 3, 3) from `models_eval/models_info.json` via `bop_toolkit_lib.misc.get_symmetry_transformations` (identity first, continuous symmetries discretised at 1 % of the diameter, for YCB-V in M8). Padded with the **identity**: a repeated identity does not change the min, so no mask is needed. On LM-O: K = 2; eggbox (10) and glue (11) have one flip each, every other object the identity only.
    - The flips in `models_info.json` are not exactly 180°: eggbox 179.8°, glue **178.5°** (scanned models). They are used as given, like the BOP19 evaluation does.
    - The symmetries' translation offsets (< 1 mm on LM-O) are ignored by the loss: it is a rotation loss.
    - `SymmetricRotation6DLoss(sym_table)` (`src/losses.py`): per sample, the smallest relative 6D Frobenius loss over `6D(R_target·S_k)`, then the batch mean. It runs in fp32 under autocast. The table is a buffer (`train.py` now moves the rotation criterion to the device).
    - Symmetries multiply on the right and the allocentric change on the left, so it works directly on the allocentric targets (a test checks this).
    - `train.py --symmetric_rotation_loss` (BOP only; needs `--rotation_format matrix --rotation_loss 6d_relative_frobenius`, else a parser error). The engine calls it with `obj_idx` in training only. Validation keeps its own symmetry-unaware `QuaternionGeodesicLoss` (symmetric validation metrics are the next bullet).
  - New `tests/test_symmetric_loss.py` (9 tests), **107 pass**:
    - Zero for every `R·S_k` (2-fold and 4-fold toy tables, with identity padding).
    - Equal to `RelativeRotation6DFrobeniusLoss` for asymmetric objects, and never above it.
    - A mixed batch uses each sample's own object.
    - Finite gradients, also exactly at a tie between two symmetric targets.
    - Allocentric invariance; fp32 output from fp16 inputs.
    - The real LM-O table.
  - Smoke in `dev` (`results/m5-symloss-smoke`, 8 objects, `token`, `domain_gap`, 224 px, 2 × 1,280 samples, `torch.compile` + AMP): runs, logs `Symmetric rotation loss: 2 symmetric objects, K = 2`. Golden SPEED after the change (`results/eval-m5-symloss-golden`): **0.0093**, byte-identical.
  - Not yet trained: the M5 exit compares eggbox/glue with and without the loss at the same config. Proposed: `token` at `domain_gap` 2× with `--symmetric_rotation_loss`, 2 seeds, against the finished `tok2x`/`tok2x2` (to confirm with the user; it can run now, in parallel with the M4 final jobs).
- [x] In-loop validation metrics on the GPU (ADD-S, MSSD, MSPD), and model selection on `(AR_MSSD+AR_MSPD)/2`. **Done 2026-10-26.**
  - New `src/bop_metrics.py`, `BOPPoseMetrics(bop_root, dataset, obj_ids, device)`:
    - Per-instance ADD (asymmetric) / ADD-S (symmetric), MSSD and MSPD of absolute poses, on the GPU, with the same definitions as `bop_toolkit_lib.pose_error`.
    - It uses **all** `models_eval` vertices (2.8k–10.2k per object, padded by repeating a vertex, masked for the ADD mean), not a subsample, so the max-based MSSD/MSPD are exact.
    - Symmetries: the BOP19 set **with** its translation offsets (`R_gt·S_k`, `R_gt·t_k + t_gt`).
    - `compute()` gives AR_MSSD, AR_MSPD (thresholds as BOP19 and `bop_evaluate.py`; MSPD scaled to 640 px width), `score = (AR_MSSD + AR_MSPD)/2`, ADD(-S)@0.1d, and per-object AR_MSSD/AR_MSPD.
    - Found while testing: bop_toolkit's ADD-S (`adi`) averages over the **GT** points the distance to the nearest estimated point. My first version went the other way (14.3 vs 12.1 mm on one eggbox instance); fixed and tested.
  - `engine.val_epoch`: with `ValidationConfig(pose_metrics=...)` it compares each prediction with the original BOP pose (`R_m2c`, `t_m2c`), as the offline evaluator does, and adds `val_metrics["bop"]`. TensorBoard `BOP/Validation/{AR_MSSD,AR_MSPD,score,ADD@0.1d}` and per object. The symmetry-unaware val loss is still computed and logged.
  - `train.py`:
    - BOP runs always compute the metrics and log a `Val BOP (n=…): AR_MSSD …, AR_MSPD …, score …, ADD(-S)@0.1d …; per object (score): …` line.
    - New `--selection_metric val_loss|bop_score` (default **`bop_score` for BOP**, `val_loss` for SPEED): `model.pth` = the epoch with the highest score.
    - `last.pt` also stores `best_score`. Resuming a pre-M5 checkpoint works (with `val_loss` selection the score is restored as −best_val_loss; SPEED behaviour unchanged).
  - Tests: new `tests/test_bop_metrics.py` (2 tests), **109 pass**:
    - Per instance, ADD/ADD-S, MSSD and MSPD equal `pose_error` within 0.02 mm/px (fp32 vs fp64), for the driller and for eggbox and glue with ~10° / 15 mm noise.
    - Flipped (symmetric) eggbox/glue predictions get MSSD = MSPD ≈ 0.
    - Per-object recalls equal `bop_evaluate.py`'s exactly, over two mixed batches.
  - **Cross-check on a real model**: `val_epoch` with `dg4x` `last.pt` on all 1,600 LM real val instances gives AR_MSSD 0.8021, AR_MSPD 0.9851, ADD(-S)@0.1d 0.6625, against the offline table's 0.8020 / 0.9849 / 0.6631. The difference is ≤ 0.0006, far inside M6's ±0.01; M6 still compares with the *official* `eval_bop19_pose.py`.
  - Cost: ~0.5 s per validation pass of 2,000 instances on the A100 (negligible next to a ~3 min epoch).
  - Smoke in `dev` (`results/m5-inloop-smoke`; 8 objects, `token`, `--symmetric_rotation_loss`, 224 px, 2 epochs, 2,000 val samples): logs `Model selection on bop_score`, the `Val BOP` line every epoch, and saves `model.pth` on a score increase.
  - The two M4 final-model jobs (`tok4x`, `tok4x2`) were launched before this change, so they still select `model.pth` on the val loss; we evaluate `last.pt` anyway.
  - **M5 training (both bullets together)**, deferred until now per the 2026-10-26 rule (train only when a decision needs it): `token`, `domain_gap` 2×, `--symmetric_rotation_loss`, 2 seeds, against `tok2x`/`tok2x2`. This is the run the M5 exit needs (eggbox/glue MSSD/MSPD improve, others within noise).
  - **Launched 2026-10-26 ~16:20**, once both code bullets were done (the keep/drop decision on the loss needs them): `lmo-m5-symtok2x-001`, `lmo-m5-symtok2x2-001` (variants `symtok2x`/`symtok2x2` in `scripts/m4_multi.sh` = `tok2x`/`tok2x2` + `--symmetric_rotation_loss`; ~16 h, done ~08:30 on 2026-10-27). Compare per object on LM real val (`eval/table_val_real.md`) with `tok2x`/`tok2x2`: eggbox (10), glue (11) MSSD/MSPD and rotation should improve, the others stay within the seed spread.
  - **Result (2026-10-27; both jobs exit 0 at 07:34 / 08:16, deleted).** LM real val, GT boxes, `last.pt`, n = 200 per object; mean of the 2 seeds:
    | | ADD(-S)@0.1d | MSSD | MSPD | rot mean (°) | trans mean (mm) |
    |---|---|---|---|---|---|
    | eggbox 10, `tok2x` → `symtok2x` | 0.938 → 0.915 | 0.804 → 0.793 | 0.989 → 0.991 | 2.71 → 2.35 | 17.4 → 18.8 |
    | glue 11 | 0.735 → 0.760 | 0.677 → 0.701 | 0.961 → 0.961 | 4.08 → 4.29 | 29.1 → 27.0 |
    | mean of 8 objects | 0.675 → 0.664 | 0.803 → 0.802 | 0.984 → 0.984 | 2.65 → 2.61 | 16.7 → 16.8 |
    - Seed spread of the plain loss (`tok2x` vs `tok2x2`): eggbox MSSD 0.023, glue 0.027, mean ADD 0.009. So the glue gain (+0.024 MSSD, +0.025 ADD-S) is about one seed spread, eggbox is flat, and the 8-object means are unchanged: **the exit criterion ("eggbox and glue improve") is not clearly met; nothing got worse.**
    - Likely reason: the plain model was not hurt by the symmetry in the first place. Its rotation errors are already measured against the closest symmetric pose, and they are small (eggbox 2.4–3.0°, glue ~4°, no 180° flips), so the network does not average the two flips. The real LM images probably break the symmetry through texture/appearance; this was not checked further.
    - **Decided with the user 2026-10-27: the symmetric loss stays on from now on** (correct by construction, costs nothing, matters more for YCB-V's continuous symmetries). The M4 final models (`tok4x`) train without it.
      - `train.py`: `--symmetric_rotation_loss` is now on by default for BOP with `--rotation_format matrix` and `6d_relative_frobenius` (`--no-symmetric_rotation_loss` turns it off; asking for it elsewhere is still an error; SPEED unchanged).
      - `scripts/m4_multi.sh` passes `--no-symmetric_rotation_loss` so its earlier variants stay as they were run; `symtok*` turn it back on.
      - Checked in `dev` (1-epoch overfit runs): default → `Symmetric rotation loss: 2 symmetric objects, K = 2`; `--no-…` → plain loss; SPEED + the flag → parser error. Tests pass.
- **Exit:** eggbox (10) and glue (11) MSSD/MSPD improve, and the other objects stay within noise.

### M6 — Official BOP evaluation with GT boxes (oracle) (W8–9)
- [x] `src/predict_bop.py` writes BOP19 CSV:
  - File name `fastposevitmo_lmo-test.csv` (no `_` inside the method name).
  - Columns `scene_id,im_id,obj_id,score,R,t,time`; R row-major, **t in mm**.
  - **Done 2026-10-28 in `src/bop_evaluate.py`** (agreed with the user: no separate `predict_bop.py`; it already wrote one BOP19 CSV per split with all objects through `bop_toolkit_lib.inout.save_bop_results`, R row-major, t in mm, score 1).
    - `--method` default is now `fastposevitmo` (was `fpvit`), checked to contain no `_` → `OUT_DIR/fastposevitmo_lmo-<split>.csv`.
    - Real `time` column (was -1): BOP's seconds to estimate all objects of an image, from `time_per_image`, a separate timed pass image by image after 3 warm-up images: crop + normalisation of every instance from the decoded image in memory, the network on that batch, R/t recovery, with CUDA sync. Reading/decoding the JPEG from the NAS is not timed. The poses still come from the batched pass.
    - Check with the M4 final model (`tok4x`, LM-O test, into `.../lmo-m4-tok4x-001/tok4x/eval_bop/`): `inout.load_bop_results` reads all 1,445 estimates; R and t identical to the M4 test CSV (max difference 0); one `time` per image for all 200 images, median **53 ms** (33–90 ms; ~7 objects per image on one A100, no detector); t in mm. 139 tests pass.
- [x] Run `eval_bop19_pose.py` and report AR_VSD, AR_MSSD, AR_MSPD and AR. **Done 2026-10-28** with the M4 final model `tok4x` (`last.pt`), GT (oracle) boxes, `scripts/bop_eval.sh` in `dev` (vispy/EGL renderer, ~1 min per split):
  | split | targets | AR | AR_VSD | AR_MSSD | AR_MSPD | time/image | our AR_MSSD / AR_MSPD (mean over instances) |
  |---|---|---|---|---|---|---|---|
  | **LM-O test** (BOP19) | 1,445 | **0.689** | 0.514 | 0.660 | 0.892 | 53 ms | 0.660 / 0.892 |
  | LM real val (= BOP LM test targets of the 8 LM-O objects) | 1,600 | 0.835 | 0.710 | 0.809 | 0.985 | 19 ms | 0.809 / 0.985 |
  - **Exit check: our MSSD/MSPD equal the official ones to ≤ 0.001 on both splits** (required ±0.01); the in-loop GPU metrics already matched the offline ones to ≤ 0.0006 (M5), so all three agree.
  - LM real val needs a targets file with only the 8 objects (otherwise the 7 other LM objects count as misses): new `scripts/bop_subset_targets.py` writes `test_targets_bop19_lmo8.json` into our tree (`$NAS_HOME/datasets/bop/lm/`, not `$DATA_DIR`); the val CSV is the `val_real` one renamed `fastposevitmo_lm-test.csv`; `bash scripts/bop_eval.sh DIR fastposevitmo_lm-test.csv --targets_filename=test_targets_bop19_lmo8.json`.
  - Outputs: `.../lmo-m4-tok4x-001/tok4x/eval_bop/` (test: CSV, `eval/fastposevitmo_lmo-test/scores_bop19.json`, `bop_eval_test.log`) and `eval_bop_val/` (val).
  - Against §2.2 (RGB-only, **real detections**, so ours with GT boxes is an upper bound for us): GDRNPP-PBR-RGB 0.713 (VSD 0.549 / MSSD 0.701 / MSPD 0.887), ZebraPose PBR 0.729, SurfEmb-PBR-RGB 0.663, CosyPose-PBR 0.633, EPOS 0.547. Our MSPD (0.892) is at the top of the RGB methods; MSSD (0.660) and VSD (0.514) are where we lose (the regressed depth). The guess "low 0.6s" in §2.2 was pessimistic: VSD is 0.146 below MSSD, as for the others, but MSPD lifts the mean. First comparable number with detections: M7.
- [ ] **Multi-GPU training (DDP)**, added 2026-10-21 at the user's request (the supervisor runs multi-GPU jobs). Placed here because from M6 on the whole pipeline exists (train → predict → official AR), so 1 vs N GPUs can be compared on the final metric.
  - Today `src/train.py` is single-GPU: one `cuda` device, no `torch.distributed`, and `SingleDeviceMuonWithAuxAdam`. A job started with `GPUS=2` leaves the second GPU idle.
  - Implement:
    - Launch with `torchrun --nproc_per_node=$GPUS` inside `rcp/train.sh`; the user says how many GPUs a run gets (`GPUS=N bash rcp/train.sh ...`, with `CPUS`/`MEMORY` scaled accordingly).
    - `DistributedDataParallel`, a `DistributedSampler` (with `set_epoch`) for train, validation on rank 0 or sharded + all-reduced.
    - Logging, TensorBoard, `model.pth` and `last.pt` on rank 0 only; resume loads on every rank; per-rank RNG states.
    - Muon's distributed variant (`MuonWithAuxAdam`) instead of `SingleDeviceMuonWithAuxAdam`.
    - Effective batch = per-GPU batch × N: check the linear LR auto-scaling (§6.7) still fits.
    - `torch.compile` and AMP under DDP.
  - Tests and checks:
    - With N = 1 the results stay identical to the single-GPU code (SPEED golden and a short smoke run).
    - A 2-GPU unit test on CPU (gloo backend) checking that ranks see disjoint data and end with identical weights.
  - Measure (one short run per setting, same total batch): images/s and GPU utilisation for 1, 2 and 4 GPUs, and the AR of 1 vs 2 GPUs at equal epochs. Keep multi-GPU as an option only if the speed-up is real (≥ 1.6× on 2 GPUs); if the M2 throughput bullet showed the run is limited by data loading from the NAS, fix that first (local copy of the PBR images), since more GPUs would just wait together.
  - Estimated effort: about 1 day.
  - **Started 2026-10-26** (independent of the running M4/M5 jobs, which use their own code copies). **Code done and tested; benchmark done: 1.30× on 2 GPUs, below the 1.6× target, so multi-GPU stays an option and training stays on 1 GPU.**
    - `src/distributed.py`:
      - `init_distributed()`: joins the process group when started by torchrun (`WORLD_SIZE > 1`; NCCL, one GPU per process, `cuda:LOCAL_RANK`), else returns world size 1.
      - `shard`, `all_reduce_mean`, `all_gather_object`, `barrier`, `cleanup_distributed`: all no-ops with world size 1.
    - `train.py`:
      - Wraps the model in `DistributedDataParallel` (before `torch.compile`).
      - **`--batch_size` stays the global batch**, split evenly over the GPUs (it must divide). The LR auto-scaling, the steps per epoch and the schedule are therefore identical for any number of GPUs, so the §6.7 LR question does not arise.
      - Data: `ObjectBalancedSampler(rank, world_size)` shards each epoch's order (every rank builds the same order and takes every N-th sample); without virtual epochs, `DistributedSampler`. Both get `set_epoch`.
      - Rank 0 only: logging (other ranks WARNING, no log file), TensorBoard, progress bars, validation (on the plain module, other ranks wait at a barrier), `model.pth` and `last.pt`.
      - Train losses are averaged over the ranks before logging. `last.pt` stores every rank's RNG state (`rng_ranks`), and resume restores each rank's own (old checkpoints still load).
      - At the end, every rank's weight checksum is logged, with a warning if they differ.
      - New log line per epoch: `Train time: … s, … samples/s [on N GPUs]`.
    - **Optimizer: kept `SingleDeviceMuonWithAuxAdam`** on every rank instead of the distributed `MuonWithAuxAdam` (a change from the plan above). DDP averages the gradients first, so every rank computes the same update; only the Newton–Schulz work is repeated. A test checks that the ranks end identical and equal to one process on the full batch. Switch only if the benchmark shows the optimizer step matters.
    - Launch: `scripts/m4_multi.sh` and the new `scripts/ddp_bench.sh` start `python -m torch.distributed.run --standalone --nproc_per_node=<GPUs of the job>` when the job has more than one GPU. `rcp/train.sh` is unchanged (it belongs to the RCP scripts): `GPUS=2 CPUS=32 MEMORY=64Gi bash rcp/train.sh NAME bash scripts/m4_multi.sh VARIANT`. `--num_workers` is per process.
    - Tests: new `tests/test_distributed.py` (4 tests), **113 pass**:
      - The shards are disjoint and of equal size, and together they are the 1-GPU epoch.
      - A real **2-process DDP run on the CPU** (gloo) with the Muon + aux Adam setup of `train.py` ends with bit-identical weights on both ranks, equal to one process on the full batch (measured gap 3.5e-10, tested at 1e-6).
      - With world size 1 the helpers do nothing.
    - **N = 1 unchanged**: with one process no DDP code runs (no `DDP:` log line, same sampler order, same loaders). Smoke runs in `dev` (`results/m6-ddp-n1-python`, `results/m6-ddp-n1-torchrun1`; 8 objects, token, 2 epochs, then resumed to 3) work with plain `python` and with `torchrun --nproc_per_node=1`. Golden SPEED (`results/eval-m6-ddp-golden`): **0.0093**, byte-identical. (Training is not seeded and uses `cudnn.benchmark` + max-autotune, so "identical" means the same code path, not bitwise-equal runs.)
    - **Speed benchmark** (`scripts/ddp_bench.sh`: the M4 setting at 384 px, token, `domain_gap`, global batch 32; 3 epochs of 16,896 samples; samples/s of epochs 2–3; `nvidia-smi` every 5 s). Launched 2026-10-26 16:20:
      | GPUs | job | CPUs / memory | samples/s | GPU util | speed-up |
      |---|---|---|---|---|---|
      | 1 | `lmo-m6-bench1-001` | 16 / 32 GiB | **216.8** | 71 % | 1.0× |
      | 2 | `lmo-m6-bench2-002` | 24 / 64 GiB, 11 workers per GPU | **280.9** | 62 % (GPU0 48 %, GPU1 54 % over the whole job) | **1.30×** |
    - 1 GPU at 71 % utilisation and 217 samples/s (vs ~257 for the GPU alone in M2): with `domain_gap` the 16 CPUs already limit the run a little, so more GPUs need proportionally more CPUs.
    - `lmo-m6-bench2-001` (32 CPUs) and `lmo-m6-bench4-001` (4 GPUs) were still Pending after 23 min (no node with 32 free CPU cores / 4 free GPUs). **Decided with the user 2026-10-26: both deleted; 4 GPUs dropped; 2 GPUs relaunched with 24 CPUs** as `lmo-m6-bench2-002` (`WORKERS=11` per process, set inside the job: `bash -c 'WORKERS=11 bash scripts/ddp_bench.sh'`, because `rcp/train.sh` only forwards `RUN_DIR`/`PYTHONPATH`). It is slightly CPU-starved next to the 1-GPU run (12 vs 16 CPUs per GPU), which also shows how CPU-bound the scaling is. Pending holds no resources, so it is left in the queue.
    - **Result (2026-10-26, job ran 17:16–17:20, deleted):** 2 GPUs give 280.9 samples/s (epochs 2–3: 280.0 / 281.8) against 216.8, i.e. **1.30×, below the 1.6× target**. The ranks ended with identical weight checksums (−14052.089 on both), so the DDP path is correct on real GPUs. After 3 short epochs the in-loop val score (n = 256) is 0.547 on 2 GPUs vs 0.563 on 1 GPU (AR_MSSD 0.365 vs 0.391), within the noise of such a short run.
    - Why so little: the GPUs are idle about half the time (48–54 %), so they wait for data. With `domain_gap` the loader is CPU-bound (1 GPU with 16 CPUs already reached only 217 of the GPU's ~257 samples/s), and this job had 24 CPUs for 2 GPUs: 281/217 ≈ 1.3 ≈ 22/16 workers. The gradient all-reduce and the repeated Muon Newton–Schulz step may cost a little too, but the run is dominated by data loading. 2 GPUs would need ~32+ CPUs on one node, which the cluster did not schedule within 23 min.
    - **Decision: multi-GPU kept as a working option (`GPUS=N`, torchrun in the scripts), not used by default.** No 2-GPU training comparison is run (it was conditional on ≥ 1.6×). To revisit only if a node with ≥ 32 free CPUs is available, or after speeding up the `domain_gap` augmentation (e.g. on the GPU).
- [x] **Single-GPU training speed-up**, added 2026-10-28 at the user's request (iterative: profile → fix the main bottleneck → re-measure, several cycles). Every cycle is checked: full test suite, SPEED golden eval (0.0093, `results.json` byte-identical), and the losses / val score of the benchmark run against the baseline's run-to-run noise.
  - Benchmark: the `scripts/ddp_bench.sh` setting on 1 GPU in `dev` (A100 80 GB, 16 CPUs, 14 workers): M4 config (384 px, token, `domain_gap`, symmetric loss, batch 32), 3 epochs × 16,896 samples, samples/s of epochs 2–3. Results in `$NAS_HOME/results/speed-c<N>-*`.
  - **Baseline (c0)**: 219 / 224 samples/s (2 runs), 144 ms/step; final train loss 0.567 / 0.575, val score (n = 256) 0.559 / 0.558.
    - Diagnosis: the data loader is *not* the limit (the step waits 0.1 ms for data; loader alone 307 samples/s; `domain_gap` costs 29 ms/sample of CPU, RandomSunFlare alone 15 ms when applied). `torch.profiler` over 12 steps: the GPU has only ~55 ms of work per step (forward + backward 23.7 ms, optimizer 28 ms), but the optimizer's per-parameter Python loop launches ~4,600 small kernels per step, so the GPU waits for the CPU.
  - **Cycle 1: batched optimizer step** (`src/optimizer.py`): the same Muon / Adam math with one `torch._foreach_*` kernel per operation for all parameters, and one batched Newton–Schulz per matrix shape (77 Muon matrices → 8 shapes). The optimizer state format is unchanged (old `last.pt` resume). New `tests/test_optimizer.py` (3): state bit-identical to the old loop, parameters bit-identical on CPU (GPU: Muon identical, Adam within 1 float32 ulp).
    - **305 / 304 samples/s (+38 %)**, 105 ms/step; loss 0.553, val score 0.551 (within noise). Golden SPEED 0.0093 byte-identical; 116 tests pass. Data wait now 5 ms/step: the loader (~307 samples/s alone) becomes the next limit.
    - Re-measured after cycle 1 (synthetic data, synced timings of each part): GPU forward + backward 73.2 ms, optimizer 18.3 ms, H2D 2.3 ms → a full GPU-only step is 91 ms (351 samples/s max). Real run 105 ms/step: the loader (~307 samples/s alone) is now the limit.
  - **Cycle 2: exact fast RandomSunFlare** (`FastRandomSunFlare` / `sun_flare_overlay_fast` in `src/bop_dataset.py`, used by `pixel_transform("domain_gap")`): the same random parameters and **bit-identical** images. The original blends the full image once per flare circle and 40 times for the sun; the flare circles now blend only their bounding box, and the 40 concentric sun blends become one per-channel lookup table T[first covering circle][value], built with albucore's own `add_weighted`. Both shortcuts rely on two properties of that function (elementwise and layout-independent; blending a pixel with itself is the identity), which a test checks. 18.2 → 6.1 ms per flare (the rest is mostly albumentations' float → uint8 conversion); `domain_gap` pipeline 20.8 → 16.0 ms per crop.
    - New `tests/test_sun_flare.py` (8): both properties; bit-identical output on float32 and uint8 images, 3 sizes and 2 sun radii; and the whole `domain_gap` pipeline bit-identical with the fast and the original transform.
    - **319 / 321 samples/s (+5 %, +45 % over the baseline)**, 100 ms/step, data wait 0.3 ms; loss 0.561, val score 0.559 (within noise). Golden SPEED 0.0093 byte-identical; 124 tests pass.
  - **Cycle 3: fewer CPU–GPU syncs** (`src/engine.py`): the progress bar used the losses only every 10th batch, but three `.item()` per batch made the CPU wait for the GPU at every step. Now one transfer of the three losses on the displayed batches only; same displayed values (new test in `tests/test_engine_batch.py`). **No measurable gain** (318 / 319 samples/s; GradScaler still syncs once per step); kept because it is correct and cheaper (also for DDP). Loss 0.554, val score 0.548.
  - **Cycle 4: fused Newton–Schulz** (`src/optimizer.py`): Muon's orthogonalisation was 15 of the optimizer's 18 ms of GPU time. `B = b·A + c·A@A` and `X = a·X + B@X` are now `torch.baddbmm` (as in the reference Muon). **Not bit-identical**, but closer to the exact iteration: against a float64 reference on the ViT-B shapes, relative error 1.1 % vs 2.3 % before (max element 2.2 % vs 6.5 %); `torch.compile` of the old code differs from it by the same amount. 15.0 → 10.3 ms for the 72 large matrices; GPU-only step 91 → 88 ms.
    - Tests (`tests/test_optimizer.py`, now 15): for 6 shapes on CPU and GPU, the new iteration is at least as close to float64 as the old one.
    - **330 / 331 samples/s (+3.5 %, +50 % over the baseline)**, 97 ms/step, GPU busy 91 %; loss 0.554, val score 0.549 (within noise). Golden SPEED 0.0093 byte-identical; 137 tests pass.
  - Tried and dropped (no gain): 10 instead of 14 workers (336 / 327 samples/s, data wait creeps up; CPU contention is not the limit); `torch.compile` modes `max-autotune-no-cudagraphs` / `default` (forward + backward 74 ms in all modes: compute-bound at ~145 TFLOPS); cycle 5, one float32 conversion + one foreach add per Muon shape group instead of 77 mixed-dtype adds (optimizer CPU issue time 7.8 → 6.8 ms, but 330 / 332 samples/s: reverted to keep the code simpler).
  - **Behaviour check over more runs** (2026-10-28; the original code re-run from a worktree at `53d2581`):
    | code | runs | samples/s (epochs 2–3) | final train loss | val score after 3 ep (n = 256) |
    |---|---|---|---|---|
    | original (`53d2581`) | 3 | 219 / 224 / 218–222 (mean 221) | 0.567 / 0.575 / 0.579 | 0.559 / 0.558 / 0.525 |
    | final (`39787a9`) | 3 | 330 / 332–334 / 331–332 (mean 332) | 0.554 / 0.551 / 0.574 | 0.549 / 0.549 / 0.521 |
    The two sets overlap completely (the val score at n = 256 after 3 epochs varies by ±0.02 between identical runs). **Result: +50 % throughput (221 → 332 samples/s; epoch of the benchmark 77 → 51 s) with unchanged training.** For a 150-epoch 4× run (`tok4x`, ~34 h) this means roughly ~23 h.
  - Where the time goes now (GPU-only step 88 ms: forward + backward 73 ms, optimizer ~15 ms; real step 97 ms). Further options, none tried, each would need its own check:
    - ~6 %: the optimizer's CPU issue time (~7 ms) becomes GPU idle time because GradScaler syncs once per step; a compiled / CUDA-graphed optimizer step (LR as a tensor) or a sync-free GradScaler step would hide it.
    - ~2 %: copy the next batch to the GPU on a side stream (2.3 ms/step now on the compute stream).
    - Changes the numerics, so a decision for the user: bf16 autocast without GradScaler (no sync, no unscale), or fewer Newton–Schulz steps.
- [x] **Multi-GPU (DDP) training speed-up**, added 2026-10-28 at the user's request (same method as the single-GPU bullet; **the single-GPU code path is not changed**: every change is in the DDP-only `DistributedMuonWithAuxAdam`, chosen by `train.py` only when the world size is > 1). Only bit-identical changes were allowed unless closer to the exact values; none needed numerics changes.
  - Tooling: `scripts/ddp_profile.py` (runs `src/train.py` unchanged and prints per rank and epoch the data wait and the optimizer's CPU time per step), `scripts/ddp_bench.sh` (now through it; `RUNS=2` repeats in one job for the noise, `WORKERS_LIST` sweeps the worker count). Jobs `lmo-m6-ddpc<N>-001` (2 × A100 SXM with NVLink NV12, 24 CPUs, 64 GiB), all deleted; outputs in their run folders. Each change: full tests, golden SPEED 0.0093 byte-identical, ranks' weight checksums identical, losses / val score vs the noise.
  - Results (samples/s of epochs 2–3, 2 runs per job; 1 GPU = 332 samples/s with the same code):
    | cycle | change | samples/s | vs 1 GPU | step | numerics | train loss / val score (n = 256) |
    |---|---|---|---|---|---|---|
    | c0 | baseline DDP (after the single-GPU speed-up) | 496 / 489 | 1.48× | 65 ms | – | 0.569 / 0.547, 0.558 / 0.553 |
    | c1 | Newton–Schulz split over the GPUs (each orthogonalises its share of every shape group, one all-gather) instead of the full NS duplicated on both | 505 / 506 | 1.52× | 63 ms | bit-identical (test: 2-process gloo, = single-device optimizer) | 0.548 / 0.540, 0.560 / 0.541 |
    | c2 | 8 instead of 11 workers per GPU (does CPU contention slow the main processes?) | 465 | 1.40× | 68 ms | config only | 0.565 / 0.563 — rejected: the loader becomes the limit (data wait 4–9 ms/step) |
    | c3 | one float32 conversion + one foreach add per shape group instead of 77 mixed-dtype adds; skip the ×1 scale | 526 / 526 | 1.58× | 61 ms | bit-identical (385/385 tensors on GPU; CPU test) | 0.559 / 0.560, 0.568 / 0.561 |
    | c4 | this rank's NS share as one CUDA graph over fixed buffers (~110 launches → 1 copy + 1 replay); optimizer CPU time 11.8 → 5.2 ms/step | 551 / 546 | **1.65×** | 58 ms | bit-identical (graph replay = eager, 10/10; GPU test vs single-device over 4 steps) | 0.561 / 0.547, 0.583 / 0.518 |
    - Overall **492 → 548 samples/s (+11 %), scaling 1.48× → 1.65×** of one GPU (≥ 1.6×, the M6 target). The data wait stays 0.1–0.2 ms/step (11 workers per GPU). Losses / val scores of all runs overlap the 1-GPU runs (0.551–0.579 / 0.518–0.559); the optimizer is bit-identical, so the differences are training noise (not seeded).
  - Diagnosis along the way: c0 GPU utilisation 83–92 %, data wait 0.2 ms; the duplicated NS (~15 ms GPU per step on each GPU) and then the optimizer's CPU issue time (13 ms under DDP vs 6 ms uncontended, because 22 workers + 2 main processes share 24 CPUs) right after GradScaler's per-step sync, during which the GPU waits.
  - Left (per GPU at 16 images: forward + backward 39.3 ms, optimizer ~9.5 ms, H2D 1.2 ms → ideal ~50 ms ≈ 640 samples/s; now 58 ms): gradient all-reduce and all-gather (~1–3 ms, NVLink), the ~5 ms optimizer CPU gap after the sync (a graphed whole optimizer step needs tensor LRs, which changes the rounding), and `gradient_as_bucket_view` (~1 %). Not tried.
  - With 1.65× now above the 1.6× threshold, the M6 decision "1 GPU by default" (made at 1.30×) could be revisited; the user chose 1 GPU for now (2026-10-28).
- **Exit:** official LM-O AR with oracle boxes, with in-loop MSSD/MSPD within ±0.01 of the official values. Multi-GPU: N = 1 identical to before, and a measured 1 vs N speed-up table in §9.

### M7 — Real detections (W9–11)
- [x] Download the BOP default detections for seen objects: [`bop23_default_detections_for_task1.zip`](https://bop.felk.cvut.cz/media/data/bop_datasets_extra/bop23_default_detections_for_task1.zip) (GDRNPPDet).
  - **Done 2026-11-02.** 50 MB zip (sha256 `f0a01aee…3249`) in `$NAS_HOME/datasets/bop_detections/zip/`, extracted to `$NAS_HOME/datasets/bop_detections/default_task1/bop23_default_detections_for_task1/` (outside `$DATA_DIR` and the repo). GDRNPPDet detections for the 7 core datasets, two variants: `gdrnppdet-pbr/` (trained on PBR only) and `gdrnppdet-pbrreal/` (PBR + real; the one BOP 2023 asks for the "default detections" award). **For LM-O both files are byte-identical** (LM-O has no real training images): `gdrnppdet-pbr/gdrnppdet-pbr_lmo-test_468338cf-….json`.
  - Format (COCO-style list): `scene_id`, `image_id`, `category_id` (= BOP obj_id), `score`, `bbox` = [x, y, w, h] in pixels (floats), `time` = the detector's seconds for that image (75–185 ms). LM-O: **38,152 detections on all 200 test images, for the 8 objects** (1: 3,403 … 8: 7,077), scores from 0.001 (median 0.003: a long low-score tail, hence "keep the top `inst_count` per object by score").
  - Time per image: the BOP 2023 page only requires a real time (not -1) and does not say whether it must include the default detector's time (checked 2026-11-02). We will report ours + the detector's `time` of that image, so it is comparable with entries that run their own detector, and say so.
- [~] Measure whether detector boxes match `bbox_visib` or `bbox_obj` better (mean IoU), and adapt the crop source and jitter. **Measured 2026-11-02; the adaptation needs a training decision (user).**
  - New `scripts/detection_iou.py` (top `inst_count` detections by score per (image, object), matched to the GT instances; output `$NAS_HOME/results/m7-detection-iou/detection_iou.md`). LM-O test, GDRNPPDet-PBR, 1,445 instances (`inst_count` is always 1 on LM-O):
    | | IoU with bbox_visib (mean / median) | IoU with bbox_obj | bbox_obj closer |
    |---|---|---|---|
    | all | 0.811 / 0.871 | **0.824 / 0.880** | 79 % |
    | visib_fract 0.1–0.5 (191) | 0.620 | **0.681** | |
    | visib_fract 0.5–0.9 (548) | 0.827 | 0.844 | |
    | visib_fract ≥ 0.9 (693) | 0.863 | 0.862 | |
    - **The detector boxes are partly amodal**: for occluded objects they lean towards bbox_obj.
    - Edges of matched detections (IoU ≥ 0.5): 94 % within ±10 % of bbox_obj (91 % of bbox_visib), p90 0.08: **the training jitter (±10 % per edge) already covers the spread**.
    - Top detection wrong (IoU_visib < 0.5) for 6.3 % of instances; eggbox 27.2 % (a better detection exists for most: only 9.4 % have none ≥ 0.5), cat 12.9 % (7.6 %), others ≤ 5 %. So the top-score choice, not the detector, loses most of these.
  - **Sensitivity of the current model** (`tok4x`, trained on bbox_visib) to the crop box: new `bop_evaluate.py --bbox_source visib|obj`, on held-out synthetic val (`val_pbr`, LM PBR scenes 48–49, occluded; LM real val is unoccluded so both boxes coincide, and LM-O test is kept for final numbers), 13,484 instances:
    | crop box | ADD(-S)@0.1d | @0.05d | MSSD | MSPD | trans mean | rot mean |
    |---|---|---|---|---|---|---|
    | bbox_visib (as trained) | **0.814** | 0.610 | **0.865** | 0.978 | 13.8 mm | 2.57° |
    | bbox_obj | 0.765 | 0.569 | 0.827 | 0.967 | 18.3 mm | 2.89° |
    The model loses 0.05 ADD / 0.04 MSSD when fed amodal boxes (mostly translation), so with detections (in between) part of that is lost. Results in `$NAS_HOME/results/m7-valpbr-bbox-{visib,obj}/`.
  - Options (training needed to choose): keep bbox_visib; train on bbox_obj; or train on a random mix of both (robust to either). Proposed: one job per candidate at the ablation budget (token, domain_gap, symmetric loss, 2×, 2 seeds; the bbox_visib baseline = the existing `symtok2x` runs), compared on `val_pbr` with both box types and on LM real val; the final check with detections on LM-O test at the end of M7.
  - **Decided with the user 2026-11-02: option 1 = the existing `symtok2x` / `symtok2x2` (bbox_visib); launch options 2 and 3 with the same configuration, only the training crop box changes.**
    - New `train.py --bop_bbox_source visib|obj|mix` (default visib, training crops only; validation stays bbox_visib for every variant). `BOPDataset(bbox_source="mix")`: bbox_visib or bbox_obj at random per training sample, bbox_visib without augmentation; the default path draws no extra random number (unchanged). Test `test_bbox_source_mix_picks_both_boxes_when_augmenting`.
    - `scripts/m4_multi.sh` variants `bobj2x`, `bobj2x2` (option 2) and `bmix2x`, `bmix2x2` (option 3) = `symtok2x` / `symtok2x2` flags + `--bop_bbox_source obj|mix`. After training, every variant is now also evaluated on `val_pbr` with both GT boxes (`eval_valpbr_visib/`, `eval_valpbr_obj/`); for `symtok2x`/`symtok2x2` run that evaluation in `dev` on their `last.pt`.
    - Same configuration as `symtok2x` (150 epochs, 2×, LR, batch, aug, symmetric loss, seeds 0/1, 1 GPU / 16 CPUs / 32 GiB); the code differs only by the M6 speed-ups, all bit-identical except the fused Newton–Schulz (closer to exact).
    - Option 1 baseline evaluated in `dev` (2026-11-02; `last.pt`, `val_pbr`, 13,484 instances, mean over instances; `.../lmo-m5-symtok2x{,2}-001/<variant>/eval_valpbr_{visib,obj}/`):
      | run | crop box at test | ADD(-S)@0.1d | @0.05d | MSSD | MSPD | trans mean | rot mean |
      |---|---|---|---|---|---|---|---|
      | `symtok2x` / `symtok2x2` | bbox_visib | 0.750 / 0.753 | 0.508 / 0.515 | 0.829 / 0.831 | 0.975 / 0.975 | 17.0 / 16.5 mm | 2.88 / 2.95° |
      | `symtok2x` / `symtok2x2` | bbox_obj | 0.693 / 0.688 | 0.468 / 0.467 | 0.784 / 0.780 | 0.961 / 0.960 | 22.0 / 22.6 mm | 3.35 / 3.35° |
      Same pattern as `tok4x` (−0.06 ADD, −0.05 MSSD with amodal boxes); seed spread ≤ 0.005.
    - **Launched 2026-11-02 ~17:20**: `lmo-m7-bobj2x-001`, `lmo-m7-bobj2x2-001` (Running), `lmo-m7-bmix2x-001`, `lmo-m7-bmix2x2-001` (Pending: no free GPU on any node). ~11 h each once running.
    - Found on the way: `test_distributed_muon_bit_identical_to_single_device` failed ~1 run in 10 (a gloo worker aborted at teardown, "terminate called without an active exception"); fixed with a `dist.barrier()` before `destroy_process_group` in the test (0 failures in 20 runs). Only the test changed.
- [x] Re-evaluate LM-O with detections; keep the top `inst_count` per object by detection score. **Done 2026-11-02 with option 1** (the M4 final model `tok4x`, trained on bbox_visib crops).
  - `bop_evaluate.py --detections FILE` (one split): for each target (image, object) the top `inst_count` detections by score (always 1 on LM-O), matched to the GT instances by IoU only for our own per-instance errors; crops around the detection box (float, converted to BOP format so `bop_bbox_to_xyxy` returns it exactly); CSV with the detection score, no estimate where nothing was detected (a miss for BOP; a failure in our tables), `time` = ours + the detector's time of the image. GT-box path unchanged (R, t, scores and table identical to the M6 run). Also fixed: `bop_evaluate.py`'s INFO logs were not printed (`basicConfig(force=True)`).
  - **LM-O test, GDRNPPDet-PBR default detections, official `eval_bop19_pose.py`** (`.../lmo-m4-tok4x-001/tok4x/eval_det/`):
    | boxes | AR | AR_VSD | AR_MSSD | AR_MSPD | time/image | ADD(-S)@0.1d (ours) |
    |---|---|---|---|---|---|---|
    | GT (M6, oracle) | 0.689 | 0.514 | 0.660 | 0.892 | 53 ms | 0.513 |
    | **default detections** | **0.668** | 0.491 | 0.641 | 0.873 | 134 ms (detector ~81 + ours ~53) | 0.473 |
    - All 1,445 targets have a detection; detections cost only 0.021 AR vs GT boxes. Our MSSD/MSPD = official again (0.641 / 0.873).
    - **Against §2.2 (RGB only, comparable now)**: ZebraPoseSAT PBR 0.729, GDRNPP-PBR-RGB 0.713, **ours 0.668**, SurfEmb-PBR-RGB 0.663, CosyPose-PBR 0.633, CDPNv2 0.624, EPOS 0.547. Ours is a direct regression from one RGB crop (no PnP, no refinement) at ~53 ms/image on an A100 for ~7 objects; our MSPD (0.873) is near the best (0.887–0.902), MSSD/VSD are where we lose (depth).
  - [x] **Open (user, 2026-11-02): also evaluate options 2 and 3 of the crop-source comparison with detections** (`bobj2x*`, `bmix2x*` vs `symtok2x*`), once their jobs are done; the first run uses option 1 (bbox_visib models).
    - **Done 2026-11-03** (user: all 6 models, options 1–3 × 2 seeds). Jobs `lmo-m7-{bobj2x,bobj2x2,bmix2x,bmix2x2}-001` finished 04:07–07:45 (exit 0) and were deleted. New `scripts/m7_det_eval.sh MODEL_DIR...` (`bop_evaluate.py --detections` + `scripts/bop_eval.sh`, same as `tok4x`; ~1.5 min per model in `dev`); output in `<variant>/eval_det/`, log `$NAS_HOME/results/m7-det-eval-crop.log`. All `last.pt`, mean over the 8 objects; GT-box columns from `eval/`, `eval_valpbr_{visib,obj}/` (ADD(-S)@0.1d):
      | option (training crops) | run | **LM-O test AR, detections** | VSD | MSSD | MSPD | ADD (det) | LM real val ADD | val_pbr ADD visib / obj |
      |---|---|---|---|---|---|---|---|---|
      | 1 bbox_visib | `symtok2x` / `symtok2x2` | **0.654 / 0.655** | 0.474 / 0.475 | 0.621 / 0.620 | 0.868 / 0.870 | 0.445 / 0.438 | 0.658 / 0.670 | 0.746 / 0.748 · 0.690 / 0.684 |
      | 2 bbox_obj | `bobj2x` / `bobj2x2` | 0.646 / 0.650 | 0.464 / 0.470 | 0.610 / 0.614 | 0.864 / 0.865 | 0.418 / 0.425 | 0.685 / 0.721 | 0.636 / 0.642 · 0.811 / 0.813 |
      | 3 mix | `bmix2x` / `bmix2x2` | 0.650 / 0.654 | 0.467 / 0.473 | 0.616 / 0.619 | 0.867 / 0.870 | 0.423 / 0.432 | 0.664 / 0.654 | 0.743 / 0.731 · 0.793 / 0.779 |
      | ref: `tok4x` (option 1, 4×) | | 0.668 | 0.491 | 0.641 | 0.873 | 0.473 | 0.672 | — |
    - **Result: with real detections the crop source barely matters** (spread 0.009 AR, seed spread up to 0.004); option 1 is marginally best (0.6545 vs 0.652 mix, 0.648 obj) and best on ADD with detections, although option 2 wins clearly on val_pbr with amodal GT boxes. So the GDRNPPDet boxes on LM-O test behave like bbox_visib for our model, not like bbox_obj (IoU 0.811 vs 0.824 was nearly equal). **Decision: keep bbox_visib (default); the final model stays `tok4x`** (4× gives +0.013 AR over the 2× runs). The training option `--bop_bbox_source` stays available.
- **Exit:** first BOP-comparable LM-O AR, compared with §2.2.

### M8 — YCB-V (W11–12)
- [x] Send Pierre the data issues from §5 before starting YCB-V. *Moved here from M0 (2026-10-20): LM-O isn't affected by them.*
  - **Done 2026-11-03:** all 5 §5 issues re-checked (still present; `train_real` has 3 scenes with 1/0/0 images), verified against the official Hugging Face file lists; message drafted and handed to Pablo, who sends it (items 1–4, without the IPD one). Our own option if `train_real` is needed: download `ycbv_train_real.zip` + `.z01` (75.7 GB) into `$NAS_HOME/datasets/` (we can't write in `$DATA_DIR`). Doesn't block M8: YCB-V trains on PBR only.
- [x] Symlink tree + index. Train on PBR only unless Pierre fixes `train_real`. There are two cameras, so per-sample K matters.
  - **Done 2026-11-03.** `bash scripts/make_bop_tree.sh ycbv` → `$NAS_HOME/datasets/bop/ycbv/` (camera_uw/cmu.json, test_targets_bop19.json, models{,_eval,_fine}, test = test_bop19, test_all, train_pbr, train_real, train_synt). The script needed no change.
  - `src/bop_dataset.py`: `SPLITS["ycbv"]` + `YCBV_OBJ_IDS` (1–21), models from `ycbv/models_eval`. **train** = `train_pbr` scenes 0–47; **val = val_pbr** = PBR scenes 48–49; **test** = `test` (BOP19 targets). **No real val:** `train_real` is broken (§5), and `test_all` holds more frames of the same 12 test videos (a leak), so it is not used. YCB-V has no `camera.json`: the image size now comes from `camera_*.json` (`_image_size`, which checks that all cameras share one size); LM-O unchanged (same cache keys).
  - Index (cached in `$NAS_HOME/cache/bop_index/ycbv_*.npz`, 15 s to build): **train 626,520 rows** from 48,000 images (23k–33k per object; LM-O: 320,959), val/val_pbr 26,349 rows from 2,000 images, **test 4,123 rows** from 900 images (= the BOP19 targets, `inst_count` always 1). 640×480 everywhere.
  - **Cameras:** PBR and all 900 test images use the `uw` intrinsics (fx 1066.8, cx 313.0, cy 241.3); `cmu` (fx 1077.8) only appears in `train_real`. K is per sample anyway.
  - Symmetric (BOP19): 1, 13 (cont.), 16, 18 (cont.), 19, 20, 21, as `models_info.json` says. Diameters 90–270 mm.
  - New `tests/test_bop_dataset_ycbv.py` (11 tests, all pass; full suite **151 passed**): test rows = BOP19 targets, PBR rows = raw annotations, row contents = raw files, reprojected models match `bbox_obj` (IoU ≥ 0.9, PBR and real test), label round trip ≤ 1e-4 on all 4,123 test targets and jittered training crops, image size from the two camera files.
  - Visual check: `vis/crops.py --dataset ycbv` (new option; LM-O figures pixel-identical) → `vis/figures/03_crops_ycbv_{train,test}.png` (2 crops per object): model points on the objects and the decoded pose on them, for train (augmented) and test. Added to `vis/run_all.sh` and `vis/README.md`.
- [x] *Found 2026-11-03:* `train.py`, `bop_evaluate.py` and `bop_metrics.py` hard-code `LMO_OBJ_IDS` / `"lmo"` (`BOP_DATASETS = ("LMO",)`, `num_objects`, symmetry table, val metrics). Generalize them to `--dataset YCBV` (objects from `SPLITS[ds]["obj_ids"]`; YCB-V has no `val_real`), plus a 21-object overfit check before the first training job.
  - **Done 2026-11-03.** `train.py --dataset YCBV`: objects, heads/tokens (21), symmetry table and GPU val metrics from `SPLITS` (`bop_config`); `--bop_val_split` and `--bop_obj_ids` are checked against the dataset (YCB-V: `val` = `val_pbr`, no `val_real`). `bop_evaluate.py --dataset lmo|ycbv` (default lmo): index, models, `num_objects`, CSV name `<method>_<dataset>-<split>.csv`, default splits `val_pbr test` for ycbv; the "(sym.)" marks in the tables now come from `models_info.json` (LM-O: still 10, 11). `bop_metrics.py` already took the dataset. 151 tests pass.
  - **LM-O unchanged:** `tok4x` re-evaluated on LM-O test with GT boxes: poses **bit-identical** to `eval_test/` and `eval_bop/` (max |ΔR| = |Δt| = 0), per-object table identical.
  - **21-object overfit check** (`$NAS_HOME/results/ycbv-m8-overfit/`, `dev`, 11 min): 256 fixed YCB-V PBR instances (seed 0, 5–23 per object, all 21 objects), no augmentation, the M4/M5 config (token, symmetric loss: `7 symmetric objects, K = 315`), batch 32, 150 epochs. Result: **ADD(-S)@0.1d 1.000**, AR_MSSD 0.995, AR_MSPD 0.982; 19 objects at score 1.000 (rotation to the nearest symmetric GT ≤ 0.15°, translation ≤ 1–4 mm). **Clamps 19 / 20 at 0.91 / 0.88:** rotation within 0.08° of the *flipped* symmetric GT and translation within 2 mm of the annotated one, but MSSD 19–21 mm, i.e. the model learned the flipped rotation without the symmetry's ~19 mm offset (see the next bullet). The pipeline itself is fine; the symmetric loss must handle the offsets before YCB-V training.
- [x] Continuous symmetries (objects 13, 18) and discrete ones (1, 16, 19, 20, 21).
  - *Found 2026-11-03:* `symmetry_table` drops the symmetries' translation offsets (< 1 mm on LM-O), but on YCB-V objects **19 and 20 (clamps) have 19.2 / 19.6 mm** (16: 1.1 mm, 21: 0.8 mm, the others 0). The symmetric loss and the GPU val metrics must use the offsets (or re-centre the models) for these objects.
  - **Done 2026-11-03.** The GPU val metrics (`bop_metrics.py`) already used the offsets; only the training loss needed the change. `symmetry_table` now returns the rotations and the offsets (n_obj, K, 3). `SymmetricRotation6DLoss.pose_loss` (used by `engine.py` whenever a table has a non-zero offset, i.e. LM-O too, with its < 1 mm) builds, on the GPU in float64, the labels of every symmetric pose (R·S_k, t + R·s_k) from the batch's BOP pose, K and crop bbox: U_k and the allocentric rotation R_rel(t_k)ᵀ·R·S_k (the allocentric change depends on the shifted translation). It takes the per-sample rotation minimum as before and passes the chosen symmetry's U_k to the translation loss. Continuous symmetries (13, 18) are discretised by bop_toolkit (K = 315) and have zero offsets. No measurable speed cost (4.5 s/epoch in the overfit check, as before).
  - Tests (`tests/test_symmetric_loss.py`, 15 pass; full suite 157): the k = 0 labels equal the dataset's labels, the k = 1 labels of the clamps equal what `BOPDataset.labels` gives to the shifted pose (≤ 1e-6, jittered crops), predicting any symmetric pose gives zero loss and that pose's U, and without offsets `pose_loss` = the rotation-only loss.
  - **Overfit check rerun** (`$NAS_HOME/results/ycbv-m8-overfit-offsets/`, same command, 11 min): **all 21 objects at score 1.000**, AR_MSSD 1.000, AR_MSPD 1.000, ADD(-S)@0.1d 1.000 (before: clamps 19 / 20 at 0.91 / 0.88). Note: LM-O training now also uses the (< 1 mm) offsets, so new LM-O runs are not bit-identical to older ones; evaluation is unchanged.
- [~] *Added 2026-11-03:* YCB-V training, the LM-O final recipe (`tok4x`: class token, `domain_gap`, symmetric loss with offsets, 150 epochs) at the LM-O 4× per-object budget (21,104 samples per object per epoch → 443,184 per epoch, 13,849 steps), 2 sampler seeds (user, 2026-11-03: no 2× comparison first). Script `scripts/m8_ycbv.sh` (train, then `last.pt` on val_pbr, test with GT boxes and test with the default detections GDRNPPDet-PBR, each test followed by the official BOP19 eval; DONE files per step).
  - Eval chain checked in `dev` with the overfit model: both test CSVs have 4,123 poses, and `scripts/bop_eval.sh` runs on YCB-V (AR 0.017, as expected for a 256-image overfit model).
  - **Jobs launched 2026-11-03 ~16:00 (CPUS=16):** `ycbv-m8-tok4x-001` (seed 0), `ycbv-m8-tok4x2-001` (seed 1; rescheduled once at start, from scratch). ~10 it/s → ~23 min/epoch → **~58 h, done ~02:00 on 2026-11-06**, then ~1 h of evaluation. Run folders `.../runs/ycbv-m8-tok4x{,2}-001/<variant>/`.
  - **2× jobs added 2026-11-03 ~16:20 (user):** `ycbv-m8-tok2x-001`, `ycbv-m8-tok2x2-001` (221,592 samples per epoch, ~29 h), same recipe, to decide whether later YCB-V runs can use 2× (on LM-O 2× ≈ 4×).
  - **GPU quota:** cvlab used 16 GPUs vs a quota of 8 at launch, so jobs above the quota are preemptible: `tok4x2` was evicted by another lab's job ~15 min after start, and the 2× jobs were Pending. Agreed with the user: keep all 4 queued (resume from `last.pt` each epoch). Finish times above assume no waiting.
  - **Done 2026-11-07** (all 4 exit code 0, deleted). Many preemptions (8–17 restarts per job, each resumed from `last.pt`; all reached epoch 150). Finished: 2× 2026-11-05 ~03:10, 4× 2026-11-06 03:41 / 06:25.
  - **Results** (`last.pt`; official BOP19 AR on YCB-V test, own recalls on val_pbr, mean over 26,349 instances):

    | Model | Test AR, detections (VSD / MSSD / MSPD) | Test AR, GT boxes | val_pbr ADD(-S)@0.1d | val_pbr MSSD / MSPD recall |
    |---|---|---|---|---|
    | `tok4x` | 0.694 (0.582 / 0.666 / 0.833) | 0.697 | 0.761 | 0.814 / 0.903 |
    | `tok4x2` | **0.702** (0.591 / 0.681 / 0.834) | **0.707** | **0.764** | **0.819 / 0.905** |
    | `tok2x` | 0.681 (0.571 / 0.658 / 0.816) | 0.687 | 0.724 | 0.788 / 0.888 |
    | `tok2x2` | 0.689 (0.586 / 0.672 / 0.808) | 0.701 | 0.718 | 0.785 / 0.886 |

  - **2× vs 4×:** with detections 4× = 0.698 vs 2× = 0.685 mean AR (+0.013; both 4× seeds above both 2× seeds; seed spread 0.007–0.009). On val_pbr the gap is clearer and consistent (MSSD +0.03, ADD +0.04). Unlike LM-O (2× ≈ 4×), **YCB-V benefits from 4×: keep 4× for YCB-V runs**, 2× still usable for quick ablations where only the ranking matters.
  - **Final YCB-V model `tok4x2`** (chosen on val_pbr, not on test): `.../runs/ycbv-m8-tok4x2-001/tok4x2/last.pt`; test AR **0.702** with detections, 0.707 with GT boxes. Detections cost only ~0.005 AR (LM-O: 0.689 → 0.668).
  - **Per-object** (`tok4x`, detections; tables in `<variant>/eval_{test,det}/table_test.md`): weakest objects 16 (wood block: MSSD recall 0.22, trans mean 90 mm), 17 (scissors: 0.38, 66 mm), 18 (large marker: 0.35), 14 (0.50), 9 (0.50, rot mean 34°: likely a near-symmetry not in BOP19), 4 (0.55). Clamps 19/20 fine (0.71 / 0.78). 16 and 17 have a large systematic translation error (median ≈ mean), worth a look.
- [x] *Added 2026-11-07 (user):* look at the weak objects 16 (wood block) and 17 (scissors) before M9.
  - **Diagnosis:** not specific to 16/17 and not a pipeline bug. On YCB-V real test **every object is predicted too far**: median +2.4 % of the distance (16: +6.7 %, 17: +5.3 %, others +0.7–3.8 %), almost all of the error along the viewing ray (perpendicular 2–5 mm). The same model has no bias on YCB-V val_pbr (+0.0 %), and `tok4x` has none on LM-O test (+0.0 %).
  - The GT poses fit the measured test depth (measured depth − GT model surface, median ≤ 0.4 mm), while our predictions sit 12–55 mm behind it. Overlaying the GT model outline on the RGB (scene 55 wood block, scene 51 scissors) shows the GT silhouette **larger than the object in the image**; our prediction hugs the real object. So on real YCB-V the RGB appearance and the GT (pose + mesh + K) disagree in scale: the objects look smaller in RGB than the GT predicts, by an amount that varies per object (mesh scale vs real object and/or RGB intrinsics/registration). An RGB-only model trained on PBR (where mesh, K and image are consistent by construction) cannot learn this offset.
  - **Decision (user, 2026-11-07): option 3, accept it.** 0.702 stays a PBR-only YCB-V result (comparable to other PBR-only entries). Fixing it needs real training images (`ycbv_train_real`, 48 GB, the missing `.z01` from the data report): train on PBR + real, or a per-object depth correction measured on real training images (never on test). Left as **open issue D8** (§8).
  - Scripts (scratchpad, not committed): depth bias per object from `eval_*/obj_*/errors_*.npz` + index; measured-depth vs model-surface check; outline overlay.
- [x] *Found 2026-11-07:* `bop_evaluate.py` writes the title "test, GT boxes" in `table_test.md` also for `--detections` runs (`eval_det/`). Cosmetic, fix the title.
  - **Done 2026-11-07:** the title names the detections file (`boxes_label`); the `eval_det/table_test.md` of the 4 YCB-V and 7 LM-O runs regenerated with `--table_only` (bodies unchanged).
- **Exit:** YCB-V AR plus a per-object table.

### M9 — Jetson / deployment (W12–13)
- [x] ONNX export with dynamic batch (`dynamic_shapes`). ONNX Runtime vs PyTorch max abs diff < 1e-3 (fp32). **Done 2026-11-08** (user go-ahead for M9 the same day, A100 only): `optimization/bop_export.py` (the SPEED script `optimization/convert_to_onnx.py` is unchanged). Graph = network only: inputs `images` (N, 3, 384, 384) normalised crops and `obj_idx` (N,) int64, output `pose` (N, 9) = merged head (t, 6D R); crop, normalisation and R, t recovery stay in Python as in `bop_evaluate.py`. Dynamo exporter, opset 22, batch dim 1–64. LM-O `tok4x` (`last.pt`, see bullet 2) → `$NAS_HOME/results/m9/lmo-tok4x/model.onnx` (330 MB, exported in < 1 min); check on 48 real LM-O test crops (GT boxes, spread over the split) at batch 1/4/8/16: **max \|ORT − PyTorch\| = 3.0e-6** (mean 4e-7; 8.1e-6 for `model.pth`; output values up to 1.0; tolerance 1e-3) → passed; report in `model.onnx.check.json`, log `export.log`. New test `tests/test_bop_export.py` (random ViT-B/224, token and heads, batch 1 and 5 ≠ the export example's 2, obj_idx not frozen), 2 passed. Findings: (1) `torch.export` fails in RCP shells without `USER` (`getpwuid(): uid not found`): the script sets `USER` itself. (2) onnxruntime-gpu 1.23 needs CUDA 12 (`libcublasLt.so.12`) and image 1.60 has CUDA 13, so its CUDA provider silently falls back to the CPU: the check ran on the CPU provider (the script now only reports `ort_cuda` when it is really active). Not needed: TensorRT is the GPU path (next bullet).
- [x] TensorRT fp16 engine with an optimization profile for 1/8/16 crops. AR change < 0.005.
  - **Done 2026-11-08**: `optimization/bop_trt.py` builds the engine (one profile, min/opt/max 1/8/16 crops, fp32 I/O; fp32 engines with TF32 off) and holds `TRTPoseModel`, a drop-in runner for `src/bop_evaluate.py --backend trt --trt_engine FILE` (own CUDA stream, larger batches in chunks of 16). `scripts/m9_trt_eval.sh` = M7 detection eval + official BOP19 eval through an engine (`DATASET=ycbv` for YCB-V). Results in `$NAS_HOME/results/m9/{lmo-tok4x,ycbv-tok4x2}/` (`eval_det_model_*`, `*.check.json`, logs).
  - **Official AR with default detections: LM-O fp16 0.6685 vs PyTorch 0.6685 (Δ 0.0000; VSD 0.4911, MSSD 0.6415, MSPD 0.8729), fp32 engine 0.6683; YCB-V fp16 0.7017 vs 0.7021 (Δ −0.0004)** → exit met. Output vs PyTorch fp32 on 48 real test crops: fp32 engine max 7.6e-6; fp16 max 2.0e-2 (LM-O) / 5.8e-3 (YCB-V), mean 4e-4, almost all in the 6D rotation values. Engine build 36 s, 167 MB (fp32: 9 s, 330 MB). Tests: `tests/test_bop_export.py` + TensorRT fp32 engine vs PyTorch with chunking (3 passed). Findings: (1) the reported AR numbers (M7 0.668, M8 0.702) come from `last.pt` (epoch 149), not `model.pth` (best val, differs by up to 1.5e-3 per weight): M9 exports `last.pt`. (2) TensorRT enables TF32 by default: an fp32 engine with TF32 gives 4e-3 max diff, without 6e-6. (3) Outputs are bit-identical between batch 1 and 16 (TensorRT picks the kernels once for the profile's opt shape). (4) The PyTorch baseline already runs under fp16 autocast, which is why fp16 TensorRT costs nothing. (5) The SPEED script `convert_to_onnx.py --create_trt_engine` no longer built: torch 2.9 writes the weights to `<model>.onnx.data` and `parser.parse(bytes)` cannot find them; fixed with `parse_from_file`. SPEED fp16 engine builds again (`results/m9/speed/`, 167 MB). The BOP `time` of these runs (LM-O 136 ms, YCB-V 116 ms per image incl. the detector) is not the latency benchmark: that is bullet 3.
- [x] Latency per image for N = 1, 4, 8, 16 objects (A100, and Jetson Orin Nano if available). The SPEED engine must still build. **Done 2026-11-08, A100 only (user: no Jetson)**: `optimization/bop_latency.py`, results `$NAS_HOME/results/m9/latency/latency.{md,json}`. Network = N crops already on the GPU (CUDA events); pipeline = one decoded LM-O test image (scene 2 im 0, 8 instances, repeated for 16) → N poses on the CPU, exactly `bop_evaluate.estimate_image` (crop, normalise, network, R, t); 30 warmup + 200 runs, median (p90 within 0.3 ms). `dev` job, 16 CPUs.

  | ms per image (median) | N = 1 | N = 4 | N = 8 | N = 16 |
  |---|---|---|---|---|
  | network, PyTorch fp32 | 11.1 | 31.6 | 56.2 | 107.5 |
  | network, PyTorch fp16 autocast (as evaluated) | 7.2 | 7.6 | 8.7 | 15.9 |
  | network, **TensorRT fp16** | **1.8** | **3.5** | **5.7** | **11.0** |
  | network, TensorRT fp32 | 9.3 | 29.9 | 55.7 | 114.9 |
  | pipeline, PyTorch fp16 autocast | 10.8 | 13.5 | 17.0 | 29.5 |
  | pipeline, **TensorRT fp16** | **5.3** | **9.3** | **14.2** | **24.0** |
  | FPS, TensorRT fp16 pipeline (pose only) | 189 | 108 | 70 | 42 |
  | FPS incl. the default detector (81.6 ms mean, GDRNPPDet, authors' GPU) | 11.5 | 11.0 | 10.4 | 9.5 |

  **Real-time claim, quantified: the pose stage runs at 70 FPS for 8 objects (14 ms) and 42 FPS for 16 on an A100 with TensorRT fp16; the whole method with the BOP default detector is ~10 FPS, limited by the detector (82 ms, 6× our 8-object pipeline)**, so an own fast detector (stretch bullet) is what a real-time on-device system needs. TensorRT fp16 vs PyTorch amp: network 1.5–4× faster (4× at N = 1, where eager PyTorch is launch-bound). fp32 TensorRT ≈ fp32 PyTorch (no gain without fp16). Two pipeline fixes found by profiling (both in `estimate_image`, so the BOP `time` of new evals drops; poses unchanged: R 8e-7, t exact vs `predict()`): (1) **normalisation on the GPU** (`normalize_crops`, = albumentations to 7e-7, test added): the albumentations Normalize + ToTensor cost 2.1 ms per crop on the CPU; TensorRT pipeline at N = 8 44 → 20.5 ms. (2) **R, t recovery on the CPU in fp32**: on the GPU its masked ops synced ~66 times per image (~9 ms fixed); 20.5 → 14.2 ms. Left: ~3.5 ms fixed + ~0.8 ms per object, mostly the OpenCV crop on the CPU (0.6 ms each; a GPU crop could remove it but changes the interpolation, so it would need an AR check). Earlier CSV `time` columns (M6–M8) keep the old, slower path's times. The SPEED engine builds (bullet 2).
- [~] (Stretch) Multi-class LW-DETR so the Jetson pipeline is self-contained. Today `object_detector/` is hard-coded to one class. *Moved here from M7 (user, 2026-11-03): it doesn't change the BOP numbers (those use the default detections), only the on-device pipeline; the default detector costs ~81 ms/image on an A100, more than our 53 ms.*
  - **Started 2026-11-08 (user: do the stretch, launch the training, M10 meanwhile).** `object_detector/bop_detector.py` (new; the SPEED one-class `object_detector/train.py` is unchanged): data = the cached BOP index grouped by image (LM-O: 47,987 PBR train images of scenes 0–47, instances with visib ≥ 0.1, bbox_visib; val = PBR scenes 48–49, COCO mAP with pycocotools), images stretched to 640², `domain_gap` pixel aug + horizontal flip. Model = LW-DETR from `object_detector/LWDETR` with the **original LW-DETR recipe** (13 query groups in training, IoU-aware BCE, loss 1 / 5 / 2, AdamW 1e-4 / encoder 1.5e-4, wd 1e-4, EMA 0.9997, grad clip 0.1, bf16), COCO weights `LWDETR_{tiny,small}_60e_coco.pth` downloaded from HF `xbsu/LW-DETR` to `$NAS_HOME/weights/lwdetr/` (all tensors but the class heads). `predict` writes BOP default-detections JSON (`category_id` = obj_id, top 100 per image, `time`) for `bop_evaluate.py --detections`. `scripts/m9_det.sh VARIANT SEED` = train → LM-O test detections → pose eval of `tok4x` with them → official BOP19 eval.
  - Prerequisites found (2026-11-08): (1) the `MultiScaleDeformableAttention` CUDA extension was never built in image 1.60 (LW-DETR did not import at all): compiled into `.venv` from `object_detector/LWDETR/lwdetr_ops` (`TORCH_CUDA_ARCH_LIST="8.0;9.0" .venv/bin/pip install --no-build-isolation .`; **a new image or venv needs this step again**). (2) Its kernel has no bf16 version: `ms_deform_attn.py` runs it in fp32 under bf16 autocast (the fp32 path is unchanged).
  - Checks in `dev`: overfit 64 PBR images (train = val, no aug, 600 iterations): mAP 0.69, AP50 0.82, still rising → boxes / labels / eval correct, DETR-slow convergence. Steady speed 0.53 s/iter GPU (batch 32, 640², tiny; group DETR matching), loader 300 ms/image with `domain_gap` at 640² (→ 22 workers). Predict + pose eval chain tested end to end with the overfit model (detections for 1,439 of 1,445 targets; detector 16 ms/image in PyTorch).
  - **Jobs launched 2026-11-08 ~10:58** (CPUS=24, 64 GiB, 50 epochs = 75k iterations; tiny ~15 min/epoch → ~12.5 h, small slower): `lmo-m9-dettiny-001` (tiny, seed 0), `lmo-m9-dettiny2-001` (tiny, seed 1, noise), `lmo-m9-detsmall-001` (small, seed 0). Next: when done, compare val_pbr mAP and LM-O pose AR with our detections vs default (0.668), build a TensorRT engine of the best and time detector + pose end to end; if it holds up, the same for YCB-V (21 classes).
  - **Tiny results (2026-11-08, both done, deleted):** LM-O test mAP 0.628 / 0.631 (AP50 0.925 / 0.930, AR100 0.703 / 0.706; default GDRNPPDet 0.657 / 0.929 / 0.721), 14.3 / 14.6 ms per image (PyTorch, A100, vs 81.6 ms default); pose AR of `tok4x` with our detections **0.645 / 0.650** vs 0.668 with the default ones. Runs `$NAS_HOME/outputs/FastPose-ViT/runs/lmo-m9-dettiny{,2}-001/tiny/`. `detsmall` at epoch 15/50 (~27 min/epoch, ~16 h left if not preempted).
  - **Results 2026-11-09 (jobs done, deleted):** `detsmall` LM-O test mAP 0.656, 16.8 ms per image, pose AR **0.654** (tiny 0.645 / 0.650, default 0.668). 224 px `tok4x` / `tok4x2` (`lmo-m10-r224tok4x{,2}-001`, LM-O test with default detections, last.pt): AR **0.643 / 0.643** vs 384 px 0.668 / 0.664 (−0.023; pipeline 7.6 vs 14.2 ms at 8 objects). Section-3 `absrot` / `absrot2`: AR **0.546 / 0.548** vs baseline symtok2x 0.655 / 0.655 (apparent rotation +0.11). Detector TensorRT engine (2026-11-09, `optimization/bop_det_trt.py`, `$NAS_HOME/results/m9/det-small-trt/`): TensorRT 10.14 GridSample is wrong for the deformable attention (engine mAP 0.633 even in fp32), fixed with a gather sampler (`GATHER_SAMPLER`, test added): fp16 engine mAP 0.657 = PyTorch, **3.3 ms** per image; full TensorRT pipeline (our detector + pose engine)
    - **LM-O AR 0.654**; end to end 17.5 ms for 8 objects (57 FPS), summed from separate timings. `docs/BOP.md` updated.
    - **Next:** re-run the joint latency (`optimization/bop_latency.py --detections $NAS_HOME/results/m9/det-small-trt/lwdetr_trt_fp16_lmo-test.json`, interrupted), monitor the 8 pending section-3 ablations + 2 T-LESS jobs (all preempted/pending on 2026-11-09 15:30), release the augmentation jobs when the queue moves.
    - **2026-11-10 00:50:** all 10 jobs Pending since 2026-11-09 08:00–16:50 ("no nodes with enough CPU cores" for 16 CPUs); user: wait until the morning, then decide (option: relaunch `scratch`/`scratch2` with CPUS=8 under new names, copying `last.pt` into the new run folder to resume).
    - **2026-11-10 12:05:** `bbox0/02/20/202`, `scratch/scratch2` done (exit 0, deleted; results not analysed yet, user); `legacy`, `legacy2`, `tless tok4x` running, `tless tok4x2` pending.
    - **18 augmentation/jitter jobs queued** (`lmo-m10-{augnone,augcolor,augblur,augnoise,augsimple,augstrong,bbox5,bbox15,bbox30}{,2}-002`, user).
    - **Section-3 ablations done 2026-11-11 (LM-O test AR, default detections, last.pt, seeds 0 / 1; baseline symtok2x 0.655 / 0.655):** absrot 0.546 / 0.548, scratch 0.516 / 0.517, legacy 0.610 / 0.625, bbox0 0.625 / 0.625, bbox20 0.650 / 0.651. Every kept choice confirmed. Per-object tables `$NAS_HOME/results/paper/section3/`; `docs/BOP.md` §2.
    - **Paper work 2026-11-11:** LaTeX draft `paper/` (Overleaf-ready); joint latency re-measured (55 FPS for 8 objects with our detector, `results/m9/det-small-trt/latency_{384,224}/`); figures `paper/figures/` (`vis/paper_{latency,qualitative,teaser}.py`); leaderboard and bibliography verified.
    - **Important finding:** the LM-O / YCB-V comparison tables were incomplete. On LM-O, PBR-only RGB methods above us include PViT-6D (0.740, ViT direct regression, one model), PoseDETR (0.695, single stage, 12 ms) and RACE-6D (0.683); on YCB-V (PBR only) we are 3rd and first among single-model methods. PViT-6D (arXiv 2311.17504) already published the `site` depth (SIZP) and a shared class-conditioned model: the paper's contributions and positioning must be agreed with the supervisor (TODOs in the draft). YCB-V depth bias quantified: 89 % of instances too far (median +20 mm) vs 48 % on LM-O.
    - **2026-11-12 07:30:** `augnone`/`augnone2` done (AR 0.626 / 0.629 vs 0.655: pixel augmentation matters, −0.028; jobs deleted). Running: `augblur{,2}` 109 / 101, `augcolor{,2}` 133 / 110, `augnoise{,2}` 86 / 75, `augsimple` 66, `bbox5` 21, T-LESS `tok4x` 49 / `tok4x2` 34 (of 150); pending (resume from last.pt): `augsimple2` 51, `augstrong{,2}` 32 / 22, `bbox15{,2}` 11 / 8, `bbox30{,2}` 3 / 0, `bbox52` 18.
    - **2026-11-12 (paper):** own-detector AR components (small, TensorRT: VSD 0.486, MSSD 0.628, MSPD 0.848, 20 ms per image; loss vs default mostly the ape's boxes) in `docs/BOP.md` and the paper; draft gained the "where we lose" paragraph and the PViT-6D comparison in related work; classic references spot-checked.
- **Exit:** latency table with the real-time claim quantified.

- **Verification audit 2026-11-14 (user request; weaknesses W1–W10, jobs prefixed `m12`):** W5 and W6 here, W1/W3/W4/W8 in M10, W2/W7 in M11, W9 in §4, W10 in §3.
  - [x] **W5, latency is narrower than it sounds** (one A100, one repeated test image, image decoding excluded, detector and pose timed separately and added; the default detector's 81.6 ms is its authors' figure): measure a true end-to-end loop, from the encoded image file to N poses (decode, detector engine, crops, pose engine, recovery, optionally the RGB-D correction), over the whole LM-O test set, per image and vs the number of detections. New `optimization/bop_e2e_latency.py`; paper / `docs/BOP.md` rows use it.
    - **Status 2026-11-14:** `optimization/bop_e2e_latency.py` written; it runs the real loop over all 200 LM-O test images with the production path (decode, detector engine, `estimate_image`, optional depth correction) and writes a BOP CSV. Its official AR equals the earlier numbers (RGB **0.654**, RGB-D **0.738**), so it is the same pipeline. **Result: 29.7 ms median per image from the encoded PNG (34 FPS) at 7.2 objects on average; PNG decode alone is 7.8 ms, detector 4.5, crops + network + pose recovery 15.0; RGB-D 30.8 ms.** The earlier "55 FPS for 8 objects" summed the medians and excluded decoding: it was optimistic. **Clean re-measure on an idle node, 2026-11-15** (`$NAS_HOME/results/m9/e2e_clean/`, 200 LM-O test images, 7.2 objects per image): **RGB 384 px 25.2 ms per image from the PNG (40 FPS)**, decode 7.8 + detector 3.4 + crops / network / recovery 14.8 (17.4 ms = 57 FPS from a decoded frame); RGB 224 px 18.5 ms (54 FPS); **RGB-D 384 px 30.0 ms (33 FPS)**, RGB-D 224 px 23.4 ms (43 FPS). (The 29.7 ms above was inflated by the busy node.) Paper (abstract, introduction, conclusion, a new measured end-to-end table, latency table labelled "summed") and `docs/BOP.md` corrected: the old 55 FPS is the decoded-frame, summed figure.
  - [x] **W6, the newer scripts have no tests** (`scripts/depth_correction.py`, `scripts/pose_ensemble.py`, `object_detector/bop_detector.py` helpers, `scripts/bop_per_object_ar.py`): add unit tests (known geometry for the depth shift, symmetric alignment for the ensemble, EMA / lr schedule / mAP for the detector).
    - **Status 2026-11-14:** `tests/test_audit_modules.py` (10 tests) + the safe-loader test: depth correction on synthetic geometry (recovers a 6 % depth error to 3 mm, ignores occluders and missing depth, batched over estimates; mutation-checked), pose ensemble (`combine` refactored out of the script, byte-identical output on the real CSVs; symmetric alignment with offsets), detector helpers (lr schedule, EMA, `to_float`, image grouping). Full suite **175 passed**.

### M10 — Write-up (W13–14), T-LESS as a stretch
- [x] *Added 2026-11-07 (user):* **one model per object** (e.g. GDRNPP's "MModel" entries: a separate network per object, the one for the detected object is used at inference) vs our shared model. Possibly a small gain.
  - Existing evidence: on LM-O test the shared `tok4x` already reaches driller ADD 0.805 vs the single-object B1 0.77 (B1 with `domain_gap`: 0.815), so for the driller per-object ≈ shared. The reference GDRNPP-PBR-RGB-MModel (§2.2: LM-O 71.3, YCB-V 71.3; verify) uses per-object models.
  - Plan: LM-O first, 8 single-object runs (`--bop_obj_ids <id>`, the M4 recipe: token not needed, `domain_gap`, symmetric loss) at the same per-object budget as `tok4x` (21,104 samples per epoch, 150 epochs; ~3 h each), then the BOP AR with detections combining the 8 models' poses, vs `tok4x` (0.668). Needs a small `bop_evaluate.py` option to pick the model per object. If it helps on LM-O, consider YCB-V (21 runs).
  - Deployment cost (from M9): N engines, no batching of crops of different objects; memory N × ~86 M parameters.
  - **Jobs launched 2026-11-07 ~22:45 (user: M10 before M9, independent):** `lmo-m10-obj{1,5,6,8,9,10,11,12}-001`, `scripts/m10_single.sh OBJ_ID` (smoke-tested in `dev`: 1-epoch run, both evals, CSV holds only that object). Recipe = `tok4x` exactly (token, `domain_gap`, **no** symmetric loss as `tok4x`, bbox_visib, 150 epochs, 21,104 samples per epoch, seed 0); each evaluates `last.pt` on LM-O test for its object with GT boxes (`objN/eval_test/`) and with the default detections (`objN/eval_det/`). ~3 h each if not preempted (6 running, 2 pending at launch). Next: delete them when done, merge the 8 `eval_det` CSVs (`fastposevitmo_lmo-test.csv`) and run `scripts/bop_eval.sh` for the official AR vs `tok4x` 0.668 (detections) and 0.689 (GT boxes), plus the per-object table vs `tok4x`'s `eval_det/table_test.md`.
  - **Result (2026-11-08; all 8 exit 0 between 01:32 and 04:39, each reached epoch 150 after 0–8 preemptions; jobs deleted).** The 8 per-object CSVs merged with `scripts/m10_merge.py` (1,445 estimates; per-image time = sum over the models) into `$NAS_HOME/results/lmo-m10-single/{eval_det,eval_test}/`, official BOP19 eval:

    | LM-O test | AR | VSD | MSSD | MSPD |
    |---|---|---|---|---|
    | `tok4x` (shared), detections | 0.668 | 0.491 | 0.641 | 0.873 |
    | **8 single-object models, detections** | **0.672** | 0.495 | 0.645 | 0.877 |
    | `tok4x` (shared), GT boxes | 0.689 | 0.514 | 0.660 | 0.892 |
    | 8 single-object models, GT boxes | 0.688 | 0.511 | 0.657 | 0.896 |

    - **No gain: +0.004 AR with detections, −0.001 with GT boxes**, both inside the run-to-run noise (M7: 2 seeds of one config differ by up to 0.004, configs by 0.01). Same total training compute as `tok4x` (8 × 21,104 = 168,832 samples per epoch).
    - Per object (own table, detections; MSSD recall shared → single): ape 0.684 → 0.677, can 0.850 → 0.850, cat 0.568 → 0.577, driller 0.862 → 0.869, duck 0.541 → 0.580, eggbox 0.263 → 0.234, glue 0.624 → 0.684, holepuncher 0.682 → 0.650. Gains on duck and glue are offset by eggbox and holepuncher: mixed, no consistent pattern.
    - The summed per-image time (0.73 s with detections) is not meaningful: it adds 8 separate evaluation processes' per-image overheads.
    - **Conclusion: keep the shared model.** One model per object costs 8× the weights (8 × ~86 M parameters), one engine per object at deployment (no batching across objects, M9) and N training jobs, for no measurable AR gain on LM-O. Not worth repeating on YCB-V (21 runs). GDRNPP's MModel gain over its SModel is not reproduced with our direct regression.
- [~] T-LESS only if M8 is done by W12.
  - **In scope (user, 2026-11-08).** Data complete on the cluster copy (PBR 50 × 1,000 images, test_primesense BOP19 20 scenes, models_eval with symmetries, test_targets_bop19, GDRNPPDet default detections). Tree `bash scripts/make_bop_tree.sh tless` (2026-11-08). `SPLITS["tless"]` + `--dataset TLESS`: train = PBR scenes 0–47 (666,063 instances, 19–25 k per object), val = val_pbr = PBR 48–49 (27,744), test = `test_primesense` targets (**6,423 instances, 1,000 images, up to 7 instances of one object per image**, D6). 30 objects, 27 symmetric (11 continuous, discretised as for YCB-V): symmetric loss `K = 315`, max offset 14 mm. Index fix (2026-11-08): T-LESS images can hold more instances of an object than the target's `inst_count` (e.g. scene 6 image 248, object 11: visib 0.114 and 0.783, inst_count 1; BOP's own visibility is stricter), so the index keeps the `inst_count` most visible ones (230 targets affected); LM-O / YCB-V test indexes unchanged (checked bit-identical), dataset tests pass (43). Overfit check (`results/tless-m10-overfit/`, `dev`, 13 min): **all 30 objects at score 1.000** (AR_MSSD 1.000, AR_MSPD 1.000, ADD(-S) 1.000). Eval chain tested with that model: detections for 6,414 of 6,423 targets, official BOP19 eval runs on `test_primesense` (AR 0.054 for the overfit model, as expected).
  - **Jobs `tless-m10-tok4x-001`, `tless-m10-tok4x2-001` launched 2026-11-08 ~13:05, on hold since ~14:10 (user: priority 3 after the detectors / 224 px and the section-3 ablations; stopped while pending, no checkpoint).
  - **Relaunched 2026-11-08 ~23:20 as `tless-m10-tok4x-002`, `tless-m10-tok4x2-002` (user: queue them now, augmentation jobs stay on hold); check they do not start ahead of the pending section-3 ablations.** (`scripts/m10_tless.sh`, 4× = 633,120 samples per epoch, 150 epochs, ~83 h each on 1 GPU; 2 GPUs with DDP would be ~50 h, not used: 1 GPU is the user's default); each ends with val_pbr, test GT boxes and test detections + official AR.
  - **Result (2026-11-15; `tok4x` finished 2026-11-14 22:45 UTC, `tok4x2` 2026-11-15 12:54 UTC; jobs finished, run folders kept):** official BOP19 AR on `test_primesense` with the default detections **0.7616 / 0.7624** (VSD 0.650 / 0.651, MSSD 0.701 / 0.701, MSPD 0.933 / 0.935; 97 / 98 ms per image with the detector), GT boxes 0.8127 / 0.8120 (VSD 0.706 / 0.705, MSSD 0.754 / 0.753, MSPD 0.978 / 0.978; 16 ms). The detector costs 0.05 AR (LM-O 0.021, YCB-V 0.005); 6,414 of 6,423 targets get a detection. Per object (seed b, detections): best 29 (0.95), weakest 2 (0.53), 22 (0.63), 1 (0.65); mean over objects 0.801. In the per-object tables the translation **means** of objects 4, 14, 15 and 22 are 5·10⁴–10⁵ mm: the 10 km pose given to undetected targets (`src/bop_evaluate.py`), not a model failure (medians 7–14 mm); the paper quotes medians. Per-object AR files: `$RUN/<seed>/eval_{det,test}/per_object_ar.md`; `scripts/bop_per_object_ar.py` now leaves T-LESS objects without LM-O names. Written up in `docs/BOP.md` §1.4, the paper (T-LESS paragraph + `tab:tless`, abstract, introduction, conclusion, supplementary S-RGB-D table) and the README table.
  - [ ] **New 2026-11-15: T-LESS leaderboard comparison** (PBR-only RGB and RGB-D entries, training data checked on each submission page, as for LM-O / YCB-V in `docs/BOP.md` §1.1–1.3): then add the second block to the paper (`\todo` in `paper/sections/experiments.tex`) and fill the "not checked yet" cell in the README table.
- [~] *Added 2026-11-08:* **resolution ablation 224 vs 384 px** (open decision D7, listed in the final tables; the M3 224 px chain runs were stopped unfinished): `tok4x` recipe exactly at `vit_b_16` (224 px), 2 seeds, vs `tok4x` / `tok4x2`. Jobs `lmo-m10-r224tok4x-001`, `lmo-m10-r224tok4x2-001` (`VIT=vit_b_16 bash scripts/m4_multi.sh tok4xN`, launched 2026-11-08 ~11:00). **When they finish (they only run the val evals of `m4_multi.sh`): LM-O test with detections (`VIT=vit_b_16 bash scripts/m7_det_eval.sh RUN/tok4x`, `RUN/tok4x2`) + official AR**, and the latency of a 224 px engine (`bop_export.py --vit_model vit_b_16`), to quantify the speed/AR trade-off. **Latency done 2026-11-08** (architecture only, from the epoch-1 checkpoint; `$NAS_HOME/results/m9/lmo-r224-latency/`): TensorRT fp16 network **1.4 / 1.7 / 2.2 / 3.9 ms**, pipeline **4.4 / 6.1 / 7.6 / 11.1 ms** for N = 1 / 4 / 8 / 16 (384 px: 5.3 / 9.3 / 14.2 / 24.0), i.e. ~2× faster from N = 8 on; ONNX vs PyTorch 1.7e-5. 224 px jobs run at 23 it/s (~4 min/epoch, ~10–11 h). The small detector runs at 1.1 s/iter (~23 h for 50 epochs).
- [~] *Added 2026-11-08 (docs/PAPER_TODO.md §3, user: launch):* **paper ablations, 10 jobs** `lmo-m10-{absrot,scratch,bbox0,bbox20,legacy}{,2}-001` (`scripts/m10_ablations.sh`): each = the 2× baseline `symtok2x` (0.655 / 0.655) with one change, 2 seeds: absolute rotation target (`--no_rotation_compensation`; `bop_evaluate.py` got the same flag), ViT from scratch, bbox jitter 0 % / 20 % (default 10), `legacy` depth with 8 objects. Each ends with LM-O test AR with detections. Smoke-tested in `dev` (1 epoch, absrot + scratch, full eval chain). Launched ~12:22, some pending on cluster capacity; ~15–20 h each once running (absrot: ~7–10 min/epoch).
  - **Augmentation study, 18 jobs (user + supervisor, 2026-11-08 ~12:50)**: same script, same baseline, 2 seeds: `augnone` (no pixel aug), `augcolor`, `augblur`, `augnoise`, `augsimple` (color + blur + noise), `augstrong` (domain_gap at p = 0.8), and bbox jitter `bbox5`, `bbox15`, `bbox30` (with `bbox0` / `bbox20` above: 0 / 5 / 10 / 15 / 20 / 30 %). New `pixel_transform` pipelines (`PIXEL_AUGS`; color = brightness-contrast / gamma / hue-saturation, blur = Gaussian / motion / defocus, noise = Gaussian / ISO; every op p = 0.5); smoke-tested (`augsimple`, 1 epoch + eval), tests pass. Jobs `lmo-m10-{augnone,augcolor,augblur,augnoise,augsimple,augstrong,bbox5,bbox15,bbox30}{,2}-001`. **On hold (user, 2026-11-08 ~14:05): all 18 stopped before their first epoch (no checkpoint lost), because of the cluster queue; user priority: (1) detectors + 224 px, (2) the 10 section-3 ablations, (3) T-LESS, (4) augmentation + jitter. Relaunch when (1)–(3) run or finish, with new names (`-002`), e.g. `for v in augnone augcolor augblur augnoise augsimple augstrong bbox5 bbox15 bbox30; do for s in "" 2; do CPUS=16 bash rcp/train.sh lmo-m10-$v$s-002 bash scripts/m10_ablations.sh $v$s; done; done`.**
  - **Cluster capacity (2026-11-08 ~13:10):** the lab `cvlab` used 26 GPUs vs a quota of 12 (fair share 23): over-quota jobs are preempted and re-queued (they resume from `last.pt`), so at 13:15 only `dettiny`/`dettiny2` ran and 33 jobs were Pending. Total queued work ≈ 700 GPU-hours (28 ablations × ~17 h, T-LESS 2 × ~83 h, 224 px ~18 h, detector small ~25 h): wall times depend on the queue. Ask the user to prioritise if the queue does not move.
- [~] Final tables (draft 2026-11-08 in `docs/BOP.md`, pending rows marked): LM-O and YCB-V vs PBR-only RGB leaderboard entries (YCB-V leaderboard read 2026-11-08, same automatic-extractor caveat as LM-O; refinement column "?" for MRC-Net, PoseDETR, SC6D: check), ablation table, deployment/latency table, limitations, reproduce commands:
  - AR on LM-O and YCB-V vs RGB-only BOP entries without refinement.
  - Ablations (depth mode, conditioning, symmetry loss, augmentation, resolution 224/384).
  - Latency.
- [~] README "BOP" section, `scripts/bop_*.sh`, merge `multi-object` into `main`. Paper guidelines `docs/PAPER_FEAT.md` and missing-pieces checklist `docs/PAPER_TODO.md` (2026-11-08, user: structure, style and features of the WACV paper + reading list and outline for the follow-up paper). README section "Multi-Object 6D Pose on BOP" added 2026-11-08 (headline AR, latency, commands, link to `docs/BOP.md`). **Merge into `main`: ask the user first** (after the pending runs are in).

**GPU budget (A100, one job at a time):** about 50–65 GPU-h for the core path, 85–100 with stretch goals. Long runs are chained with resume across 12 h jobs. Multi-GPU (DDP) is planned in M6; before that every training run uses 1 GPU.

---

- **Verification audit 2026-11-14, items in M10 (W1, W3, W4, W8):**
  - [~] **W1, the ablation baseline was trained on older code** (`symtok2x`, commit c0fd04c, 2026-10-26; later ablations ran 12 training-relevant commits newer: optimizer speed-ups, sun flare, symmetry offsets). Small effects (jitter 20 %, color −0.010) and the M11 gains (0.005–0.02) are not safe against that. Re-train the baseline on current code, 2 seeds: `scripts/m10_ablations.sh base`, jobs `lmo-m12-base{,2}-001`; the ablation tables use the new baseline for rows trained on current code.
    - **Status 2026-11-14:** `base` variant added to `scripts/m10_ablations.sh` (smoke run OK); jobs `lmo-m12-base-001` / `lmo-m12-base2-001` queued (Pending, lab over quota). When done: new-code baseline replaces symtok2x in the small-effect comparisons; decide with the validation score (W2). **Done 2026-11-15:** the re-trained baseline gives LM-O test AR 0.650 / 0.654 (mean 0.652; VSD 0.470, MSSD 0.617, MSPD 0.868) and validation score (`eval_valn`) 0.875 / 0.871, against 0.655 / 0.655 and 0.874 for the original run: **the older code changed nothing beyond the seed spread (−0.003 AR)**. The bbox 5 / 15 / 30 % and M11 rows were trained on the current code and compare with the new baseline; the paper tables keep the 0.655 baseline row and the supplementary lists both (`tab:supp:m11`).
  - [x] **W3, headline numbers are the better of two seeds** (LM-O 0.669, mean 0.666; YCB-V 0.702, mean 0.698; the seed was chosen on validation): report mean (and spread) next to the chosen seed everywhere, and state the selection rule (validation score, not test) in the paper and `docs/BOP.md`.
    - **Status 2026-11-14:** Seed selection rule stated and verified on validation: LM-O real LM val `tok4x` 0.8969 vs `tok4x2` 0.8909, YCB-V val_pbr `tok4x2` 0.8619 vs `tok4x` 0.8583 (the reported seeds). Paper (setup paragraph, table captions, "mean of the two seeds" rows: LM-O 0.666, YCB-V 0.698) and `docs/BOP.md` updated.
  - [~] **W4, single-seed claims** (8 single-object models 0.672; small detector 0.654; 4× heads 0.661; jitter 15 / 30 % seed a still running): second seed for the single-object models (`scripts/m10_single.sh OBJ 1`, jobs `lmo-m12-obj<ID>s1-001`) and for the small detector (`scripts/m9_det.sh small 1`, job `lmo-m12-detsmall2-001`); the 4× heads row (old M4 code, not used in the main paper) is replaced by the 2-seed token budget rows and kept in the supplementary labelled as one seed.
    - **Status 2026-11-14:** Jobs queued (Pending): `lmo-m12-obj{1,5,6,8,9,10,11,12}s1-001` (second seed of the 8 single-object models, `scripts/m10_single.sh OBJ 1`) and `lmo-m12-detsmall2-001` (second seed of the small detector). **2026-11-15: the 8 single-object seed-1 models are done** (merged with `scripts/m10_merge.py` in `$NAS_HOME/results/lmo-m12-single-s1/{eval_det,eval_test}`): LM-O AR with detections **0.666** (VSD 0.488, MSSD 0.633, MSPD 0.877; seed 0 0.672, mean **0.669**, shared model 0.668 / 0.664, mean 0.666) and with GT boxes 0.682 (seed 0 0.688): one shared model = eight single-object models, now with 2 seeds each. **2026-11-16: the small detector's second seed is done** (`lmo-m12-detsmall2-001`): mAP 0.653, AP50 0.932, AR100 0.727, LM-O AR **0.656** (VSD 0.487, MSSD 0.631, MSPD 0.849), seed a 0.654: the same within noise; no `\pending` cell is left; the 4× heads row is labelled one seed (older code, not used in the main paper).
  - [x] **W8, final weights exist only in run folders** on the scratch NAS (93 % full, backup status unknown): export the final checkpoints (LM-O `tok4x` / `tok4x2`, YCB-V `tok4x` / `tok4x2`, T-LESS when done, detector, 224 px) as compact weights-only files with a config JSON and sha256 (`scripts/export_release_weights.py`) into `$NAS_HOME/weights/final/` (also the base of the release); an off-NAS copy / Hugging Face upload needs the user's decision and token.
    - **Status 2026-11-14:** `scripts/export_release_weights.py`: 8 compact weights-only exports in `$NAS_HOME/weights/final/` (LM-O `tok4x`, `tok4x2` at 384 and 224 px, YCB-V `tok4x`, `tok4x2`, LW-DETR small and tiny) with config JSON, `MANIFEST.json` (sha256 of source and export, source run + git commit, results) and a README; each export verified to give bit-identical network outputs to its checkpoint; runs still training are refused (guard on the job log / TRAINED marker). Re-run it when T-LESS, the YCB-V detector and the M11 winners finish. **User made the off-NAS copy of `$NAS_HOME/weights/final/` (2026-11-14).** Re-copy after the next exports (T-LESS, YCB-V detector, M11 winners).

### M11 — Improve first (started 2026-11-14, user: positioning option (c))

Goal: close the LM-O gap to PViT-6D (0.740) / PoseDETR (0.695) before deciding the paper's
positioning, within the project constraints (RGB only, direct regression, no refinement, real
time). Our errors are mostly in depth (VSD / MSSD) and on heavily occluded objects; 2D alignment
(MSPD) is already at the level of the best methods. Each candidate = the ablation baseline
`symtok2x` (0.655 / 0.655) with one change, 2 seeds, LM-O test AR with default detections
(`scripts/m10_ablations.sh`); winners are combined into new final 4× runs.

- [x] **Phase A: all 10 jobs and the re-trained baseline done 2026-11-15.** Test AR with detections / noisy-box validation score `eval_valn` (mean of 2 seeds; fair baseline `lmo-m12-base{,2}-001` on the current code **0.652 / 0.873**, seeds 0.650 / 0.654 and 0.875 / 0.871; the old `symtok2x` 0.655 / 0.874): `ctx` 0.622 / 0.812, `ctx125` 0.625 / 0.816, `ctx15` 0.621 / 0.814 (**−0.03 on both: context does not help, A1–A3 dropped**); `occl` 0.654 / 0.876 (test +0.002, validation +0.003); `tlw` 0.659 / 0.874 (test +0.007, validation +0.001). **Decision by the validation rule (gain above the baseline seed spread of 0.004): nothing adopted**; `tlw` is the only one above the spread on test, but validation does not confirm it. Open for the user: try `tlw` + `occl` together at 2× (one more pair of jobs) or go to phase B with the current recipe. (2×, 2 seeds each; code + tests + smoke runs done 2026-11-14: `--crop_scale`,
  `--bop_occlusion_aug`, `--translation_loss_weight`, `--crop_context` also in `bop_evaluate.py`;
  164 tests pass). Jobs `lmo-m11-{ctx,ctx125,ctx15,occl,tlw}{,2}-001`:
  - [x] A1 `ctx125`: context margin, square of 1.25 × max(w, h) filled with the image (PViT-6D crops at 1.2×). **0.625 AR, validation 0.816: worse (−0.027).**
  - [x] A2 `ctx15`: same at 1.5×. **0.621, 0.814: worse.**
  - [x] A3 `ctx`: image instead of black bands, no margin (isolates the padding). **0.622, 0.812: worse; the black padding is not what limits the model (cause of the loss not investigated).**
  - [x] A4 `occl`: synthetic occlusion, 1-4 random rectangles (10-30 % of the side) erased, p = 0.5. **0.654, 0.876: within the seed spread of the baseline (0.652, 0.873): not adopted.**
  - [x] A5 `tlw`: translation loss weighted × 2 (depth emphasis). **0.659 (+0.007), validation 0.874 (+0.001): not confirmed on validation, not adopted.**
  - [x] A7 (2026-11-17) `tlwocc` at 4×, one seed (`lmo-m13-tlwocc4x-001`, `SPE=4×42208 scripts/m10_ablations.sh tlwocc`): **AR 0.669** (VSD 0.490, MSSD 0.642, MSPD 0.875), `eval_valn` **0.887** (`eval_val` LM real 0.901, PBR 0.926), against `tok4x` / `tok4x2` 0.668 / 0.664 (mean 0.666) and 0.886 / 0.880 (mean 0.883): **+0.003 and +0.004, below the +0.005 rule: the 2× gain (+0.009) shrinks at 4× and is within the seed spread (0.004). Not adopted, final models stay as they are.** One seed; `tok4x` / `tok4x2` ran older code (no effect, W1).
  - [x] A8 (2026-11-18) `tlwocc` at 4×, seed b (`lmo-m13-tlwocc4x2-001`, pairs with `tok4x2`): **AR 0.6666** (VSD 0.487, MSSD 0.640, MSPD 0.873), `eval_valn` **0.8855** (`eval_val` LM real 0.900, PBR 0.923), against `tok4x2` 0.664 / 0.880. **2-seed means at 4×: test 0.668 vs 0.666 (+0.002), validation 0.886 vs 0.883 (+0.003). Adopted under the new rule of 2026-11-18 (2-seed mean better on validation AND test, any margin): `tlw` + `occl` is part of the recipe from now on.**
  - [x] A6 `tlwocc` (**done 2026-11-16: test AR 0.664 / 0.658 (mean 0.661, +0.009; VSD 0.482, MSSD 0.630, MSPD 0.870), `eval_valn` 0.885 / 0.880 (mean 0.882, +0.009), `eval_val` LM real 0.902 / 0.896, PBR 0.909 / 0.907: passes the rule; the only M11 change adopted-able, open for the user: adopt for the final models or not; not tested at 4× or on YCB-V**): `tlw` + `occl` together (user, 2026-11-15; variant added to `scripts/m10_ablations.sh`). Jobs `lmo-m11-tlwocc{,2}-001` started 2026-11-16 00:20 CET, 2×, 2 seeds; adopt only if test AND `eval_valn` beat the baseline (0.652 / 0.873) by more than the seed spread (0.004).
- [x] **B4** (evaluation only, `dev`, `scripts/pose_ensemble.py`): average the poses of the two seed
  models (`tok4x` + `tok4x2`) at test time, symmetry-aware (align seed b to the closest symmetric
  pose of seed a, mean of t, rotation mean projected on SO(3)). *Done 2026-11-14
  (`$NAS_HOME/results/m11/ensemble/`):* LM-O **0.6689** (VSD 0.491, MSSD 0.641, MSPD 0.874) vs seeds
  0.668 / 0.664; YCB-V **0.7048** (0.596 / 0.680 / 0.838) vs 0.694 / 0.702. +0.003 / +0.007 over the
  seed mean for twice the pose time: **not kept**.
- **2026-11-18 (supervisor meeting): M11 closed; its open items continue in M15–M20 under the new scope and adoption rule (§0).**
- [ ] **Phase B** (after phase A, or alongside if the queue is short):
  - [ ] B1 ViT-L/16 at 512 px (SWAG E2E weights in torchvision), 2×; ~3-4× slower to train and run. **→ M15 (with ViT-B at 512 / 640 and ViT-H/14 at 518).**
  - [~] B2 8× budget on LM-O (2× → 4× gave +0.011), ~68 h per run. **Seed a `lmo-m13-base8x-001` done (2026-11-19): test AR 0.6728 vs `tok4x{,2}` 0.6685 / 0.6639; noisy real validation 0.8861 vs 0.8862 / 0.8796. Seed b `lmo-m13-base8x2-001` launched 2026-11-20 (recipe without `tlw` + `occl`, 8×, ~68 h); adopt if the 2-seed mean beats `tok4x` on noisy validation AND test. The "very high" 0.924 seen in the training log is the PBR (synthetic) validation score, where 4× has 0.922: not comparable with the real-image numbers. Feeds the M20 decision gate.**
  - [~] B3 YCB-V with its real training images (`train_real`, downloaded to `$NAS_HOME/datasets/ycbv_train_real/`, D8): should remove the +2.4 % depth bias; moves the result to the PBR + real leaderboard category. **User accepted it 2026-11-17**; supervisor 2026-11-18: report both PBR-only and PBR + real results (→ M17). **Checked 2026-11-18 (Pierre's warning):** the BOP datasets page says not to use the **80K synthetic `train_synt` images** of the original YCB-V ("objects in the same poses as the test images"); it has no such note for `train_real`. We never use `train_synt`, so this setup is allowed.
    - **Code done 2026-11-17:** tree link `$NAS_HOME/datasets/bop/ycbv/train_real_full` → the extracted 80 scenes (our own tree; the broken `train_real` link is untouched); `SPLITS["ycbv"]["train_mix"]` = PBR scenes 0–47 (626,520 rows) + 76 real videos (481,668 rows, 43 % of the rows; object-balanced sampling keeps this ratio per object) and `"val_real"` = four held-out uw videos (scenes 7, 21, 33, 45; 28,613 rows); the test scenes 48–59 are in neither. `train.py --bop_train_split train_mix`; `scripts/m14_ycbv_real.sh` (train, val on the real videos with GT and noisy boxes, test GT and detections, official eval); `scripts/m14_valreal_eval.sh` (the PBR-only models on the same real videos). Tests: `tests/test_bop_dataset_ycbv.py` +4 (splits disjoint from the test scenes, real rows match the raw files, reprojected models match `bbox_obj` on real images), 18 pass. Smoke training run OK.
    - **Baseline on the held-out real videos (PBR-only models, 2026-11-17; `eval_valreal` GT boxes / `eval_valrealn` noisy boxes, score (AR_MSSD + AR_MSPD)/2):** `tok2x` 0.692 / 0.668, `tok2x2` 0.704 / 0.683, `tok4x` 0.713 / 0.693, `tok4x2` 0.708 / 0.689 (n = 28,613; objects 1, 5, 9, 10, 15, 17, 18, 19, 21 do not appear in the four videos). Bug found: objects without instances gave nan means in `per_object_table`; fixed (`src/bop_evaluate.py`, test added) and the baseline summaries re-meaned with `scripts/summary_remean.py`. **The two running `ycbv-m14-real*` jobs have the old code copy: when they finish, run `scripts/summary_remean.py` on their `eval_valreal{,n}` folders before reading the scores.**
    - **Jobs:** `ycbv-m14-real2x-001` / `ycbv-m14-real2x2-001` (2×, 2 seeds, compare `tok2x{,2}` 0.681 / 0.689 AR with detections, PBR validation 0.834 / 0.832) started 2026-11-17 ~14:00 CET. Choose on the real validation (`eval_valrealn`), baseline = the PBR-only models on the same videos (`$RUN/<variant>/eval_valreal{,n}`). If it helps: 4× (`real4x{,2}`) for the final model, then RGB-D, paper rows (new category).
- [~] **Phase C: RGB-D variant** (user 2026-11-14: report RGB-D results too, in the main tables as a
  separate block; RGB stays the main result). Depth correction of the RGB estimates, training-free
  (`scripts/depth_correction.py`, runs via `scripts/m11_rgbd.sh`, results `$NAS_HOME/results/m11/rgbd_splat/`):
  - [x] C1 renderer version (`--method render`, 2026-11-14): LM-O 0.750, YCB-V 0.854; ~85 ms per image.
  - [x] C2 fast GPU version (`--method splat`, default, 2026-11-14): model points projected, 4×4-px
    z-buffer for visibility, batched over all estimates of an image; **3–4 ms per image**, same AR
    (LM-O 0.749 vs 0.750).
  - [x] C3 all available RGB results (2026-11-14): LM-O `tok4x` / `tok4x2` **0.749 / 0.752**, LM-O own
    LW-DETR + TensorRT **0.738 at 24 ms per image** (full RGB-D pipeline), YCB-V `tok4x` / `tok4x2`
    **0.849 / 0.851** (median shift −21 mm = the depth bias).
  - [x] C4 RGB-D leaderboard entries checked on their submission pages (training data, models, depth
    use): see `docs/BOP.md` §1.3. **RACE-6D's RGB-D entry and GDRNPP's "Fast" RGB-D entry use the same
    depth shift along the viewing ray: credit them (not a contribution).**
  - [x] C5 paper: RGB-D block in the LM-O and YCB-V tables, paragraph in §4.2, supplementary S9.
  - [x] C6 T-LESS RGB-D (2026-11-15, `$NAS_HOME/results/m11/rgbd_splat/tless_tok4x{,2}`): **AR 0.858 / 0.859** (VSD 0.789 / 0.790, MSSD 0.853 / 0.854, MSPD 0.931 / 0.933) from RGB 0.762 / 0.762, median shift −11 / −12 mm, 3.2 ms per image.
  - [x] C7 YCB-V RGB-D with our own detector (2026-11-16, `$NAS_HOME/results/m11/rgbd_splat/ycbv_ownsmall`, pose model `tok4x2`): **AR 0.827** (VSD 0.773, MSSD 0.885, MSPD 0.822) from RGB 0.681 (default detections: 0.702 → 0.851), median shift −21 mm, 3.4 ms per image for the correction.
  - [ ] C8 RGB-D for the final M11 models (after phase A / B).
  - [ ] C9 optional: RGB-D latency vs number of objects in `optimization/bop_latency.py` (today: mean per
    image over the test set).
- [x] ~~Final: combine the phase A / B winners into new 4× runs (LM-O ~34 h, YCB-V ~58 h), then
  update the paper (main + supplementary) and decide the positioning with the supervisor.~~ **Positioning decided 2026-11-18 (accuracy first, §0); the finals are M20.**

- **Verification audit 2026-11-14, items in M11 (W2, W7):**
  - [x] **W2, M11 choices would be made on LM-O test** (the ablation script only evaluates on test): choose on validation. Add a validation step (`eval_val/`: LM real val + PBR val, GT boxes, own score = (AR_MSSD + AR_MSPD) / 2 and ADD-S) to `scripts/m10_ablations.sh` and a script for finished runs (`scripts/m12_valeval.sh`), run it on the existing ablation models first to check that validation ranks them like test does; decide phase A / B only on validation, report test once.
    - **Status 2026-11-14:** `scripts/val_score.py`, the `eval_val/` step of `scripts/m10_ablations.sh` (smoke run OK) and `scripts/m12_valeval.sh` are in. Validation evaluation (LM real val + PBR val, 15,084 instances) of the 30 finished ablation models is running (`$RUN/<variant>/eval_val/`): **Done 2026-11-15, with a finding:** over the 31 finished ablation models, validation with perfect GT boxes ranks them poorly against the official test AR (Spearman **0.61** on LM real val, **0.40** on PBR val): it cannot see box-jitter effects (`bbox0` is the *best* model on GT-box validation, 0.918, and among the worst on test, 0.625, because test uses detections). Fix: `--box_noise 0.1` in `bop_evaluate.py` (GT boxes perturbed by up to 10 % per side, fixed seed, unit-tested) and an `eval_valn/` step (LM real val, noisy boxes) in `scripts/m10_ablations.sh` and `scripts/m12_valeval.sh`. **With noisy boxes the validation ranking has Spearman 0.934 against test AR** and orders 324 of 341 clearly different pairs (test gap > 0.01) correctly. **Rule from now on: M11 / M12 choices use `eval_valn` (score (AR_MSSD + AR_MSPD)/2 on noisy boxes), `eval_val` and PBR val are secondary; the test number is read once.** Results: `$RUN/<variant>/eval_val{,n}/val_score.txt`.
  - [x] **W7, RGB-D settings (bands 20 / 10 / 5 %, 50 points, 4-px cells) were set a priori, not validated**: sensitivity analysis on LM real val (`scripts/depth_correction_tune.py`: bands, iterations, minimum points, cell size, visibility threshold); keep the defaults or adopt better ones chosen on validation, then re-run the (cheap) RGB-D test evaluations if the settings change.
    - **Status 2026-11-14:** `scripts/depth_correction_tune.py`: 13 settings on LM real val with the official AR (VSD included). So far (9 of 13) the defaults (bands 20 / 10 / 5 %, 50 points, 4-px cells) are best or tied: AR 0.9412 for the default and for the neighbouring band / minimum-point settings, 0.9403 with a single iteration, 0.9327 with 5 % bands only (worse). **Done (all 13 settings, `$NAS_HOME/results/m12/rgbd_tune/tune.json`):** the defaults are within 0.001 AR of the best setting on every axis (best 0.9421 with visibility 10 %, 0.9417 with 2-px cells, default 0.9412; bands, minimum points and iterations are flat; only 5 % bands alone lose, 0.9327): the correction is not sensitive to its settings. **Decision: keep the defaults**; the RGB-D test numbers stand, no re-run.

### Scope change 2026-11-18 (supervisor meeting) → M15–M20

Accuracy first, cost does not matter (§0). Every candidate is decided by the **new adoption rule**:
its 2-seed mean beats the baseline's 2-seed mean on the noisy-box validation AND on test, by any
margin. **Common recipe from now on = the final recipe + `tlw` + `occl`** (adopted, M11 A8). Its
2× LM-O baseline already exists (`lmo-m11-tlwocc{,2}-001`: test 0.664 / 0.658, mean 0.661;
`eval_valn` 0.885 / 0.880, mean 0.882), so LM-O 2× candidates need no new baseline runs. Job
prefixes follow the milestone number (m12–m14 were used by audit / M11 jobs).

### M15 — Backbone and input resolution (W5–W7)
- Supervisor: try ViT-L / ViT-H and larger inputs (e.g. 640 × 640). We already use ViT-B/16 at
  384 px (not 224; 224 px lost 0.023 AR, M10). User 2026-11-18: launch all of them now, ViT-H
  included (multi-GPU).
- [x] Code: position-embedding interpolation in `src/vit_models.py` when `image_size` differs from
  the weights' size (bicubic on the patch grid, class tokens kept), with a unit test (identity at
  the native size; output shapes at 512 / 640). New variants in `scripts/m10_ablations.sh`
  (`vitb512`, `vitb640`, `vitl512`, `vith518`, all on the `tlwocc` recipe) and `torchrun` when
  `GPUS > 1` (pattern: `scripts/ddp_bench.sh`). **Done 2026-11-18:** `resize_pos_embedding` in
  `src/models.py` (applied when the pretrained weights are loaded), new backbones `vit_b_16_512`
  and `vit_b_16_640` (384 px SWAG weights), 2 tests in `tests/test_multi_object_heads.py`;
  `scripts/m10_ablations.sh` uses torchrun when the job has several GPUs (`NUM_WORKERS` per GPU).
- [x] Smoke in `dev`: memory and step time at batch 32 for ViT-L/16-512 and ViT-H/14-518; choose
  1, 2 or 4 GPUs (gradient accumulation only if memory forces it). **Done 2026-11-18 (A100 80 GB,
  after compilation):** ViT-B/16-384 10.2 it/s (reference, from the running jobs); ViT-B/16-640
  3.8 it/s (2.7× slower, ~29 h per 2× run on 1 GPU); ViT-L/16-512 2.1 it/s, 35 GB (4.9× slower,
  ~53 h on 1 GPU → 2 GPUs); **ViT-H/14-518 does not fit at batch 32 on one GPU** (out of memory
  at 71 GB) → 4 GPUs (8 crops per GPU). ViT-B/16-512 not measured (~2×).
- [~] Jobs `lmo-m15-{vitb512,vitb640,vitl512,vith518}{,2}-001` (2×, 2 seeds). **Launched 2026-11-18
  ~19:45 CET** (L on 2 GPUs, H on 4 GPUs). **2026-11-19:** seed a of `vitb512` / `vitl512` failed
  at start ("unknown variant": the variant parser read the trailing "2" of `vitb512` as seed b);
  fixed in `scripts/m10_ablations.sh` (an exact variant name wins), relaunched as
  `lmo-m15-vitb512-002`, `lmo-m15-vitl512-002`. Rough cost per
  sample vs ViT-B/384: B/512 ~2×, B/640 ~3×, L/512 ~5–6×, H/518 ~18× (estimates: B/512 ~21 h,
  B/640 ~35 h, L/512 ~38 h on 2 GPUs, H/518 ~3 days on 4 GPUs). Baseline: `tlwocc` 2×.
- [~] 2026-11-21 (user: do not discard a backbone from short runs): the 2× results above are the shortest budget, so B/512 "tied" is only a 2× verdict. **Long runs launched: `lmo-m15x4-{vitb512,vitb640,vitl512,vith518}{,2}-001` at 4× (`SPE=168832`; L on 2 GPUs, H on 4 GPUs, `NUM_WORKERS=11`).** Compare with `tlwocc4x{,2}` (test 0.6692 / 0.6666, noisy validation 0.8873 / 0.8855) under the adoption rule; 8× (baseline `base8x`) only if a backbone wins at 4×. Estimated 4× cost: B/512 ~38 h, B/640 ~58 h, L/512 ~64 h, H/518 ~6 days. The 2× jobs keep running.
- [~] 2026-11-21 (user): ViT-L/512 at **8×**, seed a: `lmo-m15x8-vitl512-001` (`SPE=337664`, 150 epochs, 2 GPUs; ~51 min/epoch, ~5.3 days uninterrupted). Seed b `lmo-m15x8-vitl5122-001` after the 4× result. ViT-L/512 at 2× is already adopted (means 0.8909 / 0.6857 vs `tlwocc` 0.8825 / 0.6606); compare 8× with `base8x{,2}` and the 4× L runs.
- [~] 2026-11-21 (user: go ahead) Best of M18 + M19: `lmo-m18-densesymx5{,2}-001` (symmetric XYZ targets + XYZ weight 5, 2×; variant `densesymx5` in `scripts/m18_dense.sh`) and the adopted refiner on PnP poses: `scripts/m19_on_dense.sh` (cnxs seed a on `lmo-m18-densesym-001` poses, iterations 0 / 1 / 2, run in `dev`, results `$NAS_HOME/results/m19_on_dense_densesym`).
- [~] 2026-11-21 (user: yes) Refiner trained for PnP poses: `lmo-m19-{cnxp,cnxq}{,2}-001` (`--noise pnp`: 60 / 30 / 10 % of 1.5° / 4° / 10°; `cnxq`: single 3° level), evaluated on the PnP poses of `lmo-m18-densesym-001`, iterations 0..4 (baseline: those poses, 0.9055 / 0.7065; `cnxs` gave +0.002 / +0.003). Retrain on `densesymx5` poses if it wins the M18 comparison.
- [~] 2026-11-22 (user: does a 4× budget separate `densex5` from `densesymx5`, tied at 2×?) Launched `lmo-m18x4-{densex5,densesymx5}{,2}-001` (ViT-B/384, `SPE=168832`, 1 GPU, ~25 h each) next to the dense ViT-L runs `lmo-m18x4-densesymx5l{,2}-001` (2 GPUs, ~64 h). 2× references: PnP val / test `densex5` 0.9122 / 0.7165, `densesymx5` 0.9121 / 0.7154. Add rows to `paper/sections/all_runs.tex` when done.
- Note for the report: LM-O crops are usually smaller than 384 px natively, so a gain at 640 px
  comes from finer patches, not from more image detail.
- Deadline: a final ViT-L / ViT-H model at 4× on YCB-V would take ~5× / ~18× the 58 h of ViT-B
  on one GPU; the M20 gate decides the budget (multi-GPU, 2× budget, or per-object models, which
  split the compute over many parallel jobs).

### M16 — One network per object, all datasets (W5–W7)
- Supervisor: re-run, and accept any improvement even if it costs 8× the weights. LM-O so far
  (M10 / W4): test 0.672 / 0.666 (mean 0.669) vs shared 0.668 / 0.664 (mean 0.666).
- [x] LM-O: evaluate the 16 existing per-object models on `eval_valn` (each on its object; merge)
  to apply the rule. **Done 2026-11-18** (`scripts/m16_single_valn.sh`, pooled with
  `scripts/m16_merge_valn.py`; log `$NAS_HOME/results/m16_lmo_single_valn.log`): validation
  **0.878 / 0.880 (mean 0.879)** vs shared `tok4x` / `tok4x2` 0.886 / 0.880 (mean 0.883); test
  0.672 / 0.666 (mean 0.669) vs 0.668 / 0.664 (mean 0.666). **Better on test, worse on
  validation (−0.004, mostly MSSD 0.775 vs 0.783): not adopted on LM-O by the rule.** Fix found
  on the way: the validation box noise was drawn over the evaluated objects only, so a
  single-object run saw other noisy boxes than the shared model; now drawn over the whole split
  and taken per instance (`noisy_boxes_for_rows`, test added; all-object runs unchanged).
- [x] Code: generalise `scripts/m10_single.sh` (dataset argument, object list, the dataset's
  default detections, per-object budget = that dataset's 4× budget / number of objects = 21,104
  samples per epoch, the recipe of that dataset's `tok4x` so the comparison is fair) and
  `scripts/m10_merge.py` (YCB-V, T-LESS). **Done 2026-11-18: `scripts/m16_single.sh DATASET OBJ
  [SEED]`** (train, validation per dataset, test GT and detections for that object;
  `scripts/m10_merge.py` was already dataset-independent). Smoke run in `dev` (YCB-V object 2, 1
  epoch) OK.
- [~] Jobs (user 2026-11-18: re-train now on all datasets; wave 1 = YCB-V seed 0, launched
  2026-11-18 ~19:45 CET; the other 81 submitted 2026-11-19 ~01:00 CET, user: not held for M15): `ycbv-m16-obj<ID>{,s1}-001` (21 × 2)
  and `tless-m16-obj<ID>{,s1}-001` (30 × 2), ~3 h each, launched in waves of ~20. Compare with
  `tok4x` / `tok4x2` (YCB-V 0.694 / 0.702; T-LESS 0.762 / 0.762) on validation (YCB-V
  `eval_valrealn`; T-LESS the real validation of M17) and test. **T-LESS baselines on the new real
  validation (2026-11-18):** `tok4x` / `tok4x2` `eval_valrealn` 0.877 / 0.879 (mean 0.878), GT
  boxes 0.910 / 0.911 (`$RUN/tok4x*/eval_valreal{,n}`).

### M17 — PBR + real training, all datasets (W5–W7)
- Supervisor: try both PBR only and PBR + real and keep them clearly apart; compare with the
  general leaderboard and report the category where we place best.
- [~] YCB-V: M11 B3 jobs `ycbv-m14-real2x{,2}-001` (run `scripts/summary_remean.py` on their
  `eval_valreal{,n}` first). If they win → 4× (`real4x{,2}`). Pierre's warning checked
  (2026-11-18): the BOP page warns against `train_synt` (same poses as test), not `train_real`.
- [x] LM-O (Pierre: LM images that are not in LM-O): `train_mix` = LM `train_pbr` + LM
  `test_all` scenes 1, 5, 6, 8, 9, 10, 11, 12 (real; only the scene's central object is annotated;
  never scene 2, which is the LM-O test sequence; BOP rule: "not a single pixel of test images").
  Note: `lmo/train` (10,504 images) is **rendered** (1,313 per object, black background), not
  real. **Done 2026-11-18:** the LM image order is not a video (consecutive views are ~8–9° apart,
  and every BOP19 image has a non-BOP19 view within ~5°), so a contiguous block would not reduce
  leakage. Validation therefore stays `val_real` (the 200 BOP19 images per object), and
  `train_mix` leaves those images out (`Source.exclude_targets`): ~1,000 real images per object.
  The real validation slightly favours real-trained models (neighbouring views ~5° away); the
  LM-O test does not. **Sampling:** by row counts LM-O would see ~3 % real crops (1,000 vs
  40,000 rows per object), so `--bop_real_fraction 0.25` draws 25 % of each object's samples from
  the real images (`ObjectBalancedSampler(real_fraction=…)`; unchanged when not set).
- [x] T-LESS: `train_mix` = `train_pbr` + `train_primesense` (real, turntable, black background),
  with a held-out real validation block (also T-LESS validation for M16). Background replacement
  for the real crops (with `mask_visib`) if the black backgrounds hurt. **Done 2026-11-18:**
  `train_primesense` = rings of 72 views (5° azimuth), 18 rings from +85° to −85° elevation (9
  upper rings for objects 19, 20); `val_real` = rings 1, 4, 7 (~74°, 42°, 10°; 216 images per
  object, `TLESS_REAL_VAL_IMAGES`), `train_mix` = PBR + the other rings. Background replacement
  is on from the start (`--bop_real_background`: the object, by its `mask_visib`, pasted on a
  random PBR image of the index), with `--bop_real_fraction 0.25`.
- [x] **Bug found and fixed 2026-11-18: T-LESS image size.** The index took (width, height) from
  `camera_primesense.json` (1280 × 1024), but the BOP T-LESS images are 720 × 540 (PBR and test)
  and 400 × 400 (`train_primesense`). Now read from each scene's first image
  (`_scene_image_size`; `INDEX_VERSION` 2, all index caches rebuilt). Effects: the own MSPD
  recalls of T-LESS (val_pbr, own test tables) used a 640/1280 instead of 640/720 pixel scale (too
  lenient), and the training box jitter was clamped to 1280 × 1024 instead of the image. **The
  official T-LESS AR (bop_toolkit, its own image size) is not affected**; LM-O and YCB-V are
  unchanged (camera.json = image size). The shared T-LESS `tok4x` models were trained with the
  old clamp (small effect: only boxes near the right / bottom border).
- [x] Scripts: `scripts/m17_real.sh DATASET VARIANT` (LM-O `real2x{,2}`; T-LESS `base2x{,2}` =
  PBR-only baseline with the same recipe, and `real2x{,2}`), the current recipe (+ `tlw` +
  `occl`), 2×; `scripts/m14_valreal_eval.sh` takes `DS=tless`. Tests: real-fraction sampler,
  LM-O `train_mix` rows (no scene 2, no BOP19 image, rows match the raw files), T-LESS image size
  and splits, background replacement (object kept, background replaced).
- [~] Jobs `lmo-m17-real2x{,2}-001` (baseline: `tlwocc{,2}` 2×), `tless-m17-base2x{,2}-001` and
  `tless-m17-real2x{,2}-001` (2×, 2 seeds; ~42 h each). **Launched 2026-11-18 ~19:45 CET.**

### M18 — Multi-head network: direct regression + masks + PnP (W6–W7)
- Supervisor: a ViT backbone with several heads (DR, PnP, masks), trained together; compare
  accuracy and speed at test time.
- [ ] Heads on one ViT: (1) the current direct-regression head (class token); (2) visible mask;
  (3) dense object coordinates (XYZ, normalised by the model extents) from a small upsampling
  decoder on the patch tokens (e.g. 24 × 24 → 64 × 64). Losses: current pose loss + BCE (mask) +
  L1 on XYZ inside the mask.
- [ ] Targets without a renderer for PBR: depth + `mask_visib` + GT pose back-projected to object
  coordinates; real images (M17): bop_toolkit renderer.
- [ ] Inference variants, each with the official AR and its time: DR only; PnP (OpenCV
  `solvePnPRansac` + LM refinement on the masked correspondences); PnP initialised at the DR pose;
  DR rotation + PnP translation.
- Questions: do the extra heads improve the DR head itself? Is PnP better than DR, mainly on depth
  (VSD / MSSD)? What does each variant cost?
- [ ] Tests: XYZ targets reproject onto their pixels with the GT pose; PnP on GT correspondences
  returns the GT pose. Jobs `lmo-m18-dense{,2}-001` (2×, 2 seeds).
- **Coding plan 2026-11-20 (user: "start coding M18"; code first, no training until all pieces are
  done and tested; one commit per bullet).** Design: dense head = 1×1 conv on the patch tokens →
  2× bilinear up + conv → resize to `--dense_size` (64) + conv → per-object (4 channels: 1 mask
  logit + 3 XYZ, like the per-object pose heads) 1×1 conv; all off by default (`--dense_heads`), so
  the SPEED / existing BOP models and checkpoints are untouched. XYZ target = (X − centre) /
  half-extent per axis of the model bounding box (models_info.json), so it lies in [−1, 1]; PnP
  undoes it with the same table.
  - [x] **M18.1 Dense targets in `BOPDataset`** (`dense_size`): `mask` (1, S, S) from `mask_visib`
    and `xyz` (3, S, S) + `xyz_valid` from depth + GT pose (X = Rᵀ(p − t)), sampled at the crop
    pixel centres (nearest), black bands outside the box zeroed as in the image. Depth scale read
    per scene (`scene_camera.json`, lazy cache: no index rebuild). Points outside the model box
    ± 15 % (depth noise, occluders) are not valid. Tests: reprojection of the targets, bands, scale.
    **Done 2026-11-20** (`tests/test_bop_dense.py`, 10 tests): points are 1.2 mm (PBR) / 3.0 mm (LM-O real) from the model surface, 98 % / 95 % of the mask pixels are valid; a wrong-pose control fails the check.
  - [x] **M18.2 Model**: `VisionTransformer` can return the patch tokens; `FastPoseViT(dense_heads=…)`
    adds the decoder and returns (pose output, {mask logits, xyz}). Tests: shapes, per-object
    selection, unchanged state dict / output without it. **Done 2026-11-20** (`tests/test_dense_head.py`, 9 tests; `FastPoseViT(dense_heads=True, dense_size=64)`, `model(x, obj_idx, return_dense=True)` → (pose output, (B, 4, S, S)); the pose output is bit-identical with or without the flag).
  - [x] **M18.3 Loss, engine, train args** (`--dense_heads --dense_size --mask_loss_weight
    --xyz_loss_weight`): total = pose loss + w_m · BCE(mask) + w_x · L1(XYZ in valid pixels);
    separate loss values in the log; validation reports them too. Overfit smoke run (`--bop_overfit`).
    **Done 2026-11-20** (`tests/test_dense_engine.py`, 4 tests; full suite 210 pass). The dense terms are in the training total only: the reported translation / rotation losses and the validation loss stay pose-only, so selection (BOP score) is unchanged; per-epoch mask BCE, XYZ L1 and (validation) mask IoU go to the log and TensorBoard (`Loss/{Train,Validation}/{Mask,XYZ}`, `Dense/Validation/MaskIoU`; under DDP the train values are rank 0's). Smoke run on `dev`, `--bop_overfit 256 --dense_heads`, 30 epochs: mask IoU 0.65 → 0.957, XYZ L1 0.49 → 0.061 (normalised units), BOP score on the same instances 0.979.
  - [x] **M18.4 `src/pnp.py`**: the four solvers (PnP RANSAC + LM; LM from the DR pose; DR rotation
    + least-squares translation; DR only), batched over instances. Tests: GT correspondences give
    the GT pose, noise / outlier robustness.
    **Done 2026-11-20** (`tests/test_pnp.py`, 11 tests). Chain test: dense targets of 30 PBR / 30 LM-O test instances → correspondences → `pnp` / `pnp_init` return the GT pose (0.000°, 0.00 mm); `dr_rot_pnp_t` keeps the given rotation. Synthetic: 0.5 px noise + 30 % outliers → < 1°, < 3 mm. `pnp_init` first used a fixed 8 px inlier threshold around the DR pose and dropped every point (DR errors of a few ° / cm are ~10 px): now an adaptive threshold (2.5 × the median error). A failed mode returns the DR pose with `fallback` set, so the failures can be counted.
  - [x] **M18.5 Evaluation**: `bop_evaluate.py --pose_mode {dr,pnp,pnp_init,dr_rot_pnp_t}` (+ time),
    `scripts/m18_dense.sh` (train + the four evaluations, validation first, test once).
    **Done 2026-11-20** (`tests/test_pose_modes_eval.py`, 8 tests; `src/pnp.py` `PoseRefiner`, `bop_evaluate.py --dense_heads --pose_mode`, the BOP `time` includes PnP, `pose_mode.json` counts the DR fallbacks). Oracle test: a network that returns the dataset's own labels and dense targets makes every mode return the GT poses through `predict()` and through the timed per-image path, so boxes, K, extents and row order are right. CLI check on the smoke checkpoint: all four modes run, PnP adds ~15 ms per image (10.7 → 25–27 ms), 0–4 of 400 instances fall back. `scripts/m18_dense.sh VARIANT[2]` = the tlwocc recipe + dense heads (variants `dense`, `densex5`, `densexl` = XYZ weight 1 / 5 / 0.2), then for each mode: validation (GT and noisy boxes), then the test with detections + official eval. **Not run: no training was started (user: code first).**
  - [~] **M18.6 Open points found while designing:** (a) **done 2026-11-20:** XYZ labels of symmetric objects
    (LM-O eggbox, glue; YCB-V, T-LESS): v1 uses the GT pose as is, so the target is ambiguous; now `--dense_symmetric_xyz` applies the symmetry the rotation loss picks to the XYZ target (X' = Sᵀ(X − s), `SymmetricRotation6DLoss.choose_symmetry` / `symmetric_xyz`; also in validation). Tests (`tests/test_symmetric_xyz.py`): equal to the dataset's back-projection with the symmetric pose (R·S, t + R·s) on LM-O eggbox / glue, YCB-V clamp 19 (shifted symmetry) and 13 (continuous), with a no-transform control; the engine helper follows the predicted rotation. Off by default, so the six queued `lmo-m18-*` jobs are the without-symmetry baseline; new variant `densesym` in `scripts/m18_dense.sh` (not launched). Smoke on eggbox + glue (overfit 256, 30 epochs): XYZ L1 0.051 with vs 0.050 without, as expected (an overfit run hides the ambiguity); (b) real training
    images (M17): depth + GT pose are slightly inconsistent, a rendered target (bop_toolkit
    renderer) would be cleaner; v1 filters by the model box; (c) loss weights and decoder size
    are untuned: variants run in parallel with repeats when the training starts.

### M19 — Render-and-compare refinement (W6–W7)
- What it does: render the object at the current pose estimate, let a network compare rendering
  and crop and predict a pose update, repeat 1–4 times. It corrects the residual error of a
  single-shot estimate, mostly depth (our weakest part). Typical BOP gains: a few AR points
  (CosyPose, GDRNPP + refiner, MegaPose), for several times the inference time.
- [x] R1 upper-bound check: CosyPose's PBR-trained refiner (public weights for LM-O, T-LESS, YCB-V;
  via `happypose`, installed in a separate venv so training jobs are not disturbed) on our poses,
  1–4 iterations, official eval. The training data stays PBR only; credit CosyPose.
  **Done 2026-11-20: no gain.** Setup in `docs/RCP_WORKFLOW.md` §11 (venv `$NAS_HOME/venvs/happypose`,
  happypose `dev`, six checkpoints, local patches), `scripts/cosypose_refine.py`, `scripts/m19_r1.sh`,
  results `$NAS_HOME/results/m19_r1_<dataset>/summary.txt`. Test AR with detections, our PBR-only
  poses (LM-O `tlwocc4x` seed a, YCB-V / T-LESS `tok4x`), Panda3D renderer, refiner iterations 0 / 1 /
  2 / 3 / 4: **LM-O 0.669 / 0.616 / 0.584 / 0.567 / 0.557; YCB-V 0.694 / 0.693 / 0.694 / 0.689 /
  0.689; T-LESS 0.762 / 0.743 / 0.731 / 0.722 / 0.719**; +0.04 s (1 iteration) to +0.15 s (4) per
  image. **Check that the setup is sound (`scripts/m19_refiner_check.py`: ground-truth poses + noise
  in, 20-30 images, symmetric objects left out):** on YCB-V with large noise (15°, 40 mm) the refiner
  works (rotation median 9.3° → 7.8° → 4.8°, translation 60 → 27 → 21 mm after 1 / 2 iterations), but
  with small noise (5°, 10 mm) it does nothing (3.1° → 3.3°, 15 → 17 mm; Bullet renderer, the one
  the checkpoints were trained with: the same). So the public refiner stops at about 3° / 15 mm,
  which is the accuracy of our poses already; on LM-O it makes the rotation worse even from noisy
  ground truth (5°, 10 mm: 2.2° → 6.8° Panda3D, → 11.9° Bullet), probably the vertex-coloured LM
  models, not solved. **Conclusion: this refiner is no upper bound for us; R2 (our own refiner,
  trained on perturbations of our poses) is the only way to test whether refinement helps. Decision
  for the user / supervisor (costly: batched GPU renderer + training).**
- [~] R2 our own refiner (FastPose-ViT on crop + rendering at the current pose → pose update,
  trained on PBR with perturbed poses; needs a batched GPU renderer). **User 2026-11-20: yes, build it**
  (R1 gave no gain, so R2 is the only way to know whether refinement helps us). **Plan (one commit per bullet):**
  design = a ViT-B (our `FastPoseViT`, object token) on 10 channels: the image crop around the
  projection of the object at the current pose (box × 1.4, as CosyPose), plus a render of the model
  at that pose (shaded colour 3, normalised object coordinates 3, mask 1). Output 9 values: image-space
  shift (Δu, Δv in crop sides), depth ratio exp(δz) and a 6D rotation update applied in the camera frame
  (CosyPose's parametrisation). Loss: point-matching L1 in mm / diameter, minimum over the object's
  symmetries, disentangled (full + rotation-only + translation-only). Training: PBR images with GT
  poses perturbed by a mixture of noise levels (DR-like errors and larger); inference iterates 1–4 times
  from our DR poses (CSV in, CSV out, official eval, as R1).
  - [x] **M19.1 `src/render.py`**: pure-PyTorch batched point-splatting renderer (z-buffer with
    `scatter_reduce`, 3×3 splats), model points sampled from the BOP meshes (vertex colours; YCB-V
    texture; grey T-LESS CAD) + cache. Tests: the render of a model at the GT pose has the depth of
    the PBR depth image and the mask of `mask_visib` (unoccluded instances).
    **Done 2026-11-20** (`tests/test_render.py`, 4 tests): one pixel per point + hole closing gives a silhouette within 0.5 px of the analytic sphere (3×3 splats fattened it by 1.7 px); on LM PBR instances the render has IoU > 0.93 with `mask_visib` and median depth error < 2 mm; deterministic; 9 ms for 64 renders of 224 px (13 ms at 384 px).
  - [x] **M19.2 `src/refiner.py`**: the network, the update, the symmetric disentangled point loss.
    Tests: the exact update recovers GT, loss zero at GT and invariant to symmetries, gradients.
    **Done 2026-11-20** (`tests/test_refiner.py`, 9 tests; new backbone `vit_b_16_256` = the 384 px SWAG weights at 256 px). Noise levels of the starting poses: 45 % (3°, 1 % crop side, 1.5 % depth), 35 % (8°, 3 %, 4 %), 20 % (20°, 8 %, 10 %), around our DR errors (median 4.6°, ~16 mm).
  - [x] **M19.3 data + training**: `RefineDataset` (perturbed pose → crop → sample), `src/train_refiner.py`
    (AdamW, bf16, object-balanced virtual epochs, `last.pt` resume), overfit smoke on `dev`.
    **Done 2026-11-20** (`src/refiner.py` `RefineDataset`, `src/train_refiner.py`; 500 samples/s on one A100: an 84k-sample epoch ~3 min). Findings: (1) overfitting 256 instances reaches loss 0.0008 (the chain crop → render → update → loss is consistent; `scripts/m19_look_samples.py` shows the GT render outline on the objects). (2) **A ViT-B (10-channel first layer, object token) does not learn on the real distribution: its output stays at the identity** (gain 0.0 % after 12 short epochs) with image-plus-render, zero and negated render-filter inits, with and without pixel augmentation, single or mixed noise. (3) **A ConvNeXt-B (10-channel stem, per-object heads) learns at once: 32.7 % of the starting error removed after 6 short epochs of 25,600 samples.** So the refiner's backbone is `convnext_base` (the comparison of two aligned images wants convolutions, as CosyPose's EfficientNet). Starting-pose error: the loss is ~0.11 diameters on average (depth and rotation noise like our DR errors).
  - [x] **M19.4 inference + evaluation**: `scripts/refine_bop.py` (our results CSV in, refined CSV per
    iteration out), `scripts/m19_refiner.sh` (train + eval LM-O validation/test).
    **Done 2026-11-20:** instead of a separate script the refiner is a hook of `src/bop_evaluate.py` (`--refine_weights W --refine_iterations N`): validation tables, BOP CSV, detections and the BOP `time` (refinement included) all work unchanged; test `tests/test_pose_modes_eval.py` (a stub refiner through `predict` and the timed per-image path). `scripts/m19_refiner.sh VARIANT[2]` trains and evaluates 0..4 iterations (0 = the baseline through the same code) on the poses of `lmo-m13-tlwocc4x-001`.
  - [~] **M19.5 jobs**: `lmo-m19-{cnxa,cnxb,cnxs}{,2}-001` launched 2026-11-20 ~21:30 UTC (1 GPU each, ~6 h training + ~1 h evaluation once running): ConvNeXt-B refiner, lr 5e-5 / 1e-4 / 5e-5 with small-error-weighted starting noise, 2 seeds; then 0..4 iterations on the poses of `lmo-m13-tlwocc4x-001` (noisy-box validation + test AR with detections, official eval). A smoke refiner (6 short epochs, no augmentation) is not yet useful on real images (validation 0.900 → 0.886 after 2 iterations on 3 objects, +33 ms): the long runs decide. Adopt by the rule against iteration 0 (same code path) if the 2-seed mean beats it on validation AND test.

### M20 — Final models and EPFL report (W7–W11)
- [ ] Re-examine older candidates with the new rule (evaluation-only table over existing runs):
  heads + token "both" (2× test 0.660 / 0.661 vs token 0.659 / 0.651), seed ensemble (M11 B4:
  LM-O +0.003, YCB-V +0.007 on test; validation to add), 8× budget (B2). Optional: test-time
  augmentation (average of several jittered crops), evaluation only.
- [ ] **Decision gate Fri 2026-12-04:** final recipe per dataset (backbone, heads / PnP,
  per-object, refinement, real images), within the GPU time left before the deadline.
- [ ] Finals W8–W9 (Dec 7–20): 3 datasets × {PBR only, PBR + real} × 2 seeds; RGB-D (C8);
  leaderboard tables in both categories, including the T-LESS block.
- [ ] Report W9–W11 on the `paper/` draft (main + supplementary → ~20 pages; authors Pablo Sarró
  Sánchez, Mathieu Salzmann), sent by **2027-01-01**. The publication version (+ Pierre Ancey) is
  cut from it later.

## 8. Open decisions and questions

| ID | Question | Decide at |
|---|---|---|
| D1 | Detections: BOP defaults only, or also our own multi-class LW-DETR? | M7 |
| D2 | Object conditioning: per-object heads vs class tokens (vs both) | M4 |
| D3 | Crop source for training: `bbox_visib` vs `bbox_obj` | M2 / M7 |
| D4 | `visib_fract` threshold for training instances (default 0.1) | M2 |
| D5 | BOP pixel augmentation: `standard` vs `domain_gap`; black bands vs `crop_context`. **Measured 2026-10-22: `domain_gap` better on real images** (LM-O test ADD@0.1d 0.815 vs 0.76–0.77; LM real val 0.975 vs ≤ 0.895) despite the scary preview; `crop_context` not tested yet | M3 |
| D6 | Several instances of the same object (T-LESS) | M10 |
| D7 | Backbone resolution for real time: 224 vs 384 | M3 / M9 |
| D8 | **Data being prepared (2026-11-08): the user asks the supervisor to add `train_real` to the cluster copy; meanwhile the missing part `ycbv_train_real.z01` (48.3 GB, HF `bop-benchmark/ycbv`) was downloaded to `$NAS_HOME/datasets/ycbv_raw/`, joined with the cluster `.zip` (27.4 GB, symlinked, unchanged) into a temporary `ycbv_train_real_full.zip` (75.7 GB, deleted after the check) and extracted to `$NAS_HOME/datasets/ycbv_train_real/train_real/` (1,257,337 files, 80 scenes; `unzip` refuses joined split archives, so a `zipfile` script with the disk-0 offsets corrected, kept as `$NAS_HOME/datasets/ycbv_raw/unzip_split.py`; it also recomputes zipfile's per-entry end offsets for the overlap check). **Verified 2026-11-08 13:57: all 1,257,336 files present with the archive's sizes, 80 scenes (000000–000091), each with scene_gt / scene_gt_info / scene_camera.** The `.z01` is kept in `ycbv_raw/`. NAS free space 14 TB (90 % used). Training with it is not decided yet.** Earlier **(user, 2026-11-07):** complete YCB-V with `train_real` (download `ycbv_train_real`, 48 GB incl. the missing `.z01`, into `$NAS_HOME/datasets/`) and train PBR + real, to reduce the real-image domain gap: on YCB-V real test every object is predicted ~2.4 % too far (16 wood block +6.7 %, 17 scissors +5.3 %; see M8). Until then YCB-V numbers are PBR-only | future |

| D9 | **Scope (supervisor, 2026-11-18): accuracy first.** Of three framings (small architecture with equivalent results; cheap and fast direct regression with equivalent results; beat other methods in accuracy), only the third: cost and size do not matter. PnP, refinement, per-object models, bigger backbones are allowed (§0) | Decided |
| D10 | **Adoption rule (user, 2026-11-18):** 2-seed mean better on noisy-box validation AND test, any margin (§0) | Decided |
| D11 | **Training data (supervisor, 2026-11-18):** PBR only and PBR + real, both tried and labelled; one general leaderboard with both categories, report where we place best | Decided |
| D12 | **Deliverable (supervisor, 2026-11-18):** EPFL report by 2027-01-01, no page limit (~20 pages), authors Pablo Sarró Sánchez + Mathieu Salzmann; publication later with Pierre Ancey (conference page limit). Time comparisons only if the report were framed on speed: it is not | Decided |
| D13 | **Release (supervisor, 2026-11-18):** not required; if done, code on GitHub and weights on Hugging Face | Decided |

**For the supervisor:**
- [ ] Are IPD / XYZ-IBD / HB, downloaded on 2026-10-19, meant for this project?
- [ ] GPU-hour budget.
- [x] Access to a Jetson Orin Nano. *2026-11-18: not for the semester project (setup and quantisation take too long; only worth it for a speed framing).*
- [x] Expected deliverable (report, paper, presentation). *2026-11-18: EPFL report by 2027-01-01 (D12).*

---

## 9. Results log

| Date | Run / folder | Dataset | Config | Metrics | Notes |
|---|---|---|---|---|---|
| 2026-09-30 | `results/eval-pretrained-vit-b-16-384` | SPEED test | HF `vit_b_16_384.pth`, best config | E_T 0.0221 m, E_R 0.0073 rad, score **0.0093** | Matches paper |
| 2026-09-30 | `results/smoke-train` | SPEED val | 2 epochs, README config | val score 0.1379 | First smoke run |
| 2026-10-13 | `results/eval-rcp-check` | SPEED test | same as above, RCP `dev` | score **0.0093** | Golden file |
| 2026-10-14 | `outputs/FastPose-ViT/runs/smoke-02` | SPEED val | 2 epochs, `train` job | val score 0.1400 | RCP `train` works |
| 2026-10-14 | `outputs/FastPose-ViT/runs/speed-short-001` | SPEED val | 3 epochs | val score 0.1021 | ~1.8 min/epoch |
| 2026-10-20 | `outputs/FastPose-ViT/runs/lmo-m0-access-004` | LM PBR / LM-O models | `scripts/check_bop_access.py` | `BOP ACCESS OK (lmo)` | BOP readable from `train` jobs |
| 2026-10-20 | `results/eval-m0-venv-golden` | SPEED test | same as golden, `.venv/bin/python` | score **0.0093** | `results.json` byte-identical to golden |
| 2026-10-20 | `results/bop-m0-gtpred-wrapper` | LM-O test (BOP19 targets) | GT poses as predictions, `scripts/bop_eval.sh` | AR **1.0** (VSD 1.0, MSSD 1.0, MSPD 1.0) | Eval pipeline OK; 53 s in `dev` |
| 2026-10-20 | `results/eval-m1-refactor` | SPEED test | golden command after the M1 refactor | score **0.0093** | `results.json` byte-identical to golden |
| 2026-10-20 | `results/smoke-m1-refactor` | SPEED val | `smoke-02` command, 2 epochs, `dev` | val score 0.1317 | vs 0.1400 (smoke-02); within noise |
| 2026-10-20 | `results/smoke-m1-resume` | SPEED val | `smoke-02` command, 3 epochs, SIGKILL after epoch 1 + `--resume auto` | val score 0.2137 / 0.1403 / **0.1005** | Resume continues LR and losses; 3-ep uninterrupted ref 0.1021 |
| 2026-10-21 | `results/eval-m3-image160` + `lmo-m3-image160-001` | LM-O test (GT as prediction) + SPEED val | Image 1.60 checks: BOP eval in `dev` and in a train job, golden eval | AR **1.0** (dev and train); SPEED **0.0093** byte-identical | Native libEGL, no workaround; 57 tests pass |
| 2026-10-21 | `results/lmo-m3-overfit` | 256 fixed LM PBR driller instances (train = val) | Overfit check, best SPEED config, batch 32, 150 epochs, no aug | translation 0.43 mm, rotation 0.017° (train loss 0.0015) | Pipeline learns its labels end to end; 10.5 min in dev |
| 2026-10-21 | `outputs/FastPose-ViT/runs/lmo-m3-b1-001/eval` | LM real val (n=200) / LM-O test (n=200), driller (obj 8), GT boxes | B1: ViT-B/16-384, 150 ep, `site` depth, `standard` aug | rot 1.72° / 1.61° (mean); trans 13.1 / 19.0 mm; ADD@0.1d **0.87 / 0.77**; recall MSSD 0.93 / 0.87, MSPD 0.98 / 0.93 | First B1 numbers. Rotation < 5° for 100 % / 96.5 %; translation < 5 mm only 24 % / 6 %. Val→test gap is mostly translation (occlusion in LM-O). No VSD yet |
| 2026-10-22 | `outputs/FastPose-ViT/runs/lmo-m3-abl-{base,base2,legacy,dec,domgap}-001/<variant>/eval` | LM real val / LM-O test, driller, GT boxes (n = 200 each) | M3 ablations at B1's setting (384 px, 150 ep): 2 repeats of B1, legacy depth, decoupled loss, domain_gap aug | test ADD@0.1d: base 0.760, base2 0.770 (B1 0.765), legacy 0.715, dec 0.760, **domgap 0.815**; test trans mean 18.3 / 18.7 / 20.9 / 18.0 / **17.5** mm | Noise ≈ ±0.005 ADD, ±0.4 mm; site depth confirmed; decoupled = noise; domain_gap helps sim-to-real. Full table in M3 |
| 2026-10-25 | `outputs/FastPose-ViT/runs/lmo-m4-{std,std2,dg,dg2,dg2x,dg4x}-001/<variant>/eval/obj_*` | LM real val, all 8 LM-O objects, GT boxes (n = 200 each) | M4 shared model, per-object heads, object-balanced epochs, 384 px / 150 ep: `standard` ×2 seeds, `domain_gap` ×2 seeds (1× = B1 compute), `domain_gap` 2× and 4× | mean ADD(-S)@0.1d: std 0.559 / 0.574, dg 0.631 / 0.630, dg2x 0.655, **dg4x 0.663**; driller 0.785 / 0.800 / 0.875 / 0.845 / **0.925** / 0.905 (B1 0.870); MSSD mean 0.744–0.802 | Exit criterion met on val; lowest object duck (0.29–0.43). Curves still falling at 150 ep. Per-object table in M4 |
| 2026-10-26 | `outputs/FastPose-ViT/runs/lmo-m4-{dg2x,dg2x2,tok2x,tok2x2,both2x,both2x2}-001/<variant>/eval/table_val_real.md` | LM real val, all 8 LM-O objects, GT boxes (n = 200 each) | M4 conditioning ablation at domain_gap 2×, 384 px / 150 ep, 2 seeds each: per-object heads (a), per-object class token (b), both | mean ADD(-S)@0.1d: heads 0.655 / 0.664, **token 0.679 / 0.670**, both 0.660 / 0.661; MSSD 0.795–0.806; trans 16.4–17.3 mm | Token slightly best (+0.015, ~2× seed spread); a + b = a. Table in M4 |
| 2026-10-26 | `outputs/FastPose-ViT/runs/lmo-m6-bench{1-001,2-002}/bench` | LM PBR train (3 × 16,896 samples) + LM real val (n = 256) | M6 DDP speed benchmark: M4 setting (384 px, token, `domain_gap`, global batch 32), 1 GPU / 16 CPUs vs 2 GPUs / 24 CPUs | **216.8 vs 280.9 samples/s = 1.30×**; GPU util 71 % vs 62 %; val score after 3 ep 0.563 vs 0.547 | Below the 1.6× target (loader CPU-bound); ranks bit-identical. DDP kept as an option only |
| 2026-10-27 | `outputs/FastPose-ViT/runs/lmo-m5-symtok2x{,2}-001/<variant>/eval/table_val_real.md` | LM real val, all 8 LM-O objects, GT boxes (n = 200 each) | M5 exit: `tok2x`/`tok2x2` + `--symmetric_rotation_loss` (2 seeds) | mean ADD(-S)@0.1d 0.658 / 0.670 (plain 0.679 / 0.670); MSSD eggbox 0.800 / 0.785 (plain 0.815 / 0.792), glue 0.705 / 0.697 (plain 0.690 / 0.663) | No measurable gain: glue +1 seed spread, eggbox flat, others within noise. Table in M5 |
| 2026-10-28 | `outputs/FastPose-ViT/runs/lmo-m4-tok4x{,2}-001/<variant>/eval/table_val_real.md` | LM real val, all 8 LM-O objects, GT boxes (n = 200 each) | M4 final: token, `domain_gap`, 4×, 384 px / 150 ep, 2 seeds | mean ADD(-S)@0.1d **0.672** / 0.654; MSSD 0.809 / 0.797; MSPD 0.985 / 0.985; driller 0.940 / 0.950 (B1 0.870) | = `dg4x` and `tok2x` within noise; `tok4x` proposed as the final model. Table in M4 |
| 2026-10-28 | `outputs/FastPose-ViT/runs/lmo-m4-tok4x-001/tok4x/eval_test` | **LM-O test**, all 8 objects, GT boxes (1,445 instances) | M4 final model `tok4x` (`last.pt`): token, `domain_gap`, 4×, 384 px / 150 ep | mean ADD(-S)@0.1d **0.511**, MSSD **0.653**, MSPD **0.891**; driller 0.805 (B1 0.765), duck 0.189, eggbox 0.428 | M4 exit met. Val→test −0.16 ADD (occlusion); eggbox worst (129 mm mean trans). Recalls, not AR: official AR in M6 |
| 2026-10-28 | `outputs/FastPose-ViT/runs/lmo-m4-tok4x-001/tok4x/eval_bop{,_val}` | **LM-O test** (1,445) / LM real val (1,600, 8 objects), GT boxes | M6 official BOP19 eval (`eval_bop19_pose.py`, VSD rendered) of the M4 final model `tok4x` | **AR 0.689** (VSD 0.514, MSSD 0.660, MSPD 0.892), 53 ms/image / AR 0.835 (0.710 / 0.809 / 0.985) | First official number. Own MSSD/MSPD = official to ≤ 0.001 (M6 exit). Oracle boxes: upper bound until M7 |
| 2026-10-28 | `results/speed-*`, `outputs/FastPose-ViT/runs/lmo-m6-ddpc{0..4}-001` | LM PBR train (3 × 16,896 samples), M4 config | M6 training speed-ups: 1 GPU (`dev`, 16 CPUs) and 2 GPUs (24 CPUs, NVLink); every change bit-identical except the fused Newton–Schulz (closer to exact) | 1 GPU **221 → 332 samples/s** (+50 %); 2 GPUs **492 → 548** (1.48× → **1.65×** of one GPU) | Losses / val scores within run noise; golden SPEED 0.0093 byte-identical. Tables in M6 |
| 2026-11-02 | `outputs/FastPose-ViT/runs/lmo-m4-tok4x-001/tok4x/eval_det` | **LM-O test**, BOP default detections (GDRNPPDet-PBR) | M7: `tok4x` (option 1, bbox_visib crops), top-1 detection per target, official BOP19 eval | **AR 0.668** (VSD 0.491, MSSD 0.641, MSPD 0.873), 134 ms/image incl. detector | First BOP-comparable number: above SurfEmb (0.663) and CosyPose (0.633), below GDRNPP (0.713) / ZebraPose (0.729) among RGB-only PBR methods. GT boxes 0.689 |
| 2026-11-03 | `results/ycbv-m8-overfit` | 256 fixed YCB-V PBR instances, all 21 objects (train = val) | M8 overfit check: token, symmetric loss, batch 32, 150 epochs, no aug, `dev` | ADD(-S)@0.1d **1.000**, AR_MSSD 0.995, AR_MSPD 0.982; 19 objects score 1.000, clamps 19 / 20 0.91 / 0.88 | `--dataset YCBV` works end to end; clamps fail on the ignored symmetry offset (~19 mm), M8 bullet 3 |
| 2026-11-03 | `results/ycbv-m8-overfit-offsets` | same | Same overfit check, symmetric loss with the translation offsets (`pose_loss`) | AR_MSSD **1.000**, AR_MSPD **1.000**, ADD(-S)@0.1d 1.000; **all 21 objects 1.000** | Offset fix works (clamps 0.91 / 0.88 → 1.000) |
| 2026-11-07 | `runs/ycbv-m8-tok{4x,4x2,2x,2x2}-001` | YCB-V train_pbr (all 21 objects) → val_pbr, test | M8: LM-O final recipe (token, domain_gap, symmetric loss with offsets), 4× (443,184/epoch) and 2×, 2 seeds each, 150 epochs | Test AR with detections: 4× **0.694 / 0.702**, 2× 0.681 / 0.689 (GT boxes 0.697 / 0.707, 0.687 / 0.701); val_pbr MSSD 0.814 / 0.819 vs 0.788 / 0.785 | **YCB-V AR 0.702** (`tok4x2`); 4× better than 2× on YCB-V |
| 2026-11-03 | `outputs/FastPose-ViT/runs/lmo-m5-symtok2x{,2}-001`, `lmo-m7-{bobj2x,bobj2x2,bmix2x,bmix2x2}-001` (`<variant>/eval_det`, `eval_valpbr_{visib,obj}`) | **LM-O test**, default detections; val_pbr GT boxes | M7 crop-source ablation: training crops bbox_visib (opt 1) / bbox_obj (2) / mix (3), token, `domain_gap`, 2×, 150 ep, 2 seeds each | **AR 0.654 / 0.655** (opt 1), 0.646 / 0.650 (2), 0.650 / 0.654 (3); val_pbr ADD with bbox_obj boxes 0.69 / **0.81** / 0.79 | Crop source doesn't matter with detections (≤ 0.009 AR); keep bbox_visib, final model `tok4x`. Table in M7 |

| 2026-11-08 | `results/lmo-m10-single/{eval_det,eval_test}`, `outputs/FastPose-ViT/runs/lmo-m10-obj{1,5,6,8,9,10,11,12}-001` | **LM-O test**, default detections / GT boxes | M10: one single-object model per LM-O object, `tok4x` recipe and per-object budget (21,104/epoch, 150 ep), CSVs merged | **AR 0.672** (VSD 0.495, MSSD 0.645, MSPD 0.877) / 0.688 (0.511 / 0.657 / 0.896) | = shared `tok4x` (0.668 / 0.689) within noise: keep the shared model |
| 2026-11-08 | `results/m9/lmo-tok4x` | 48 LM-O test crops (GT boxes) | M9 bullet 1: ONNX export of `tok4x` (opset 22, dynamic batch), ONNX Runtime CPU vs PyTorch fp32 (TF32 off) | max \|diff\| **3.0e-6**, mean 4e-7, at batch 1/4/8/16 | Exit < 1e-3 met. ORT CUDA provider unusable on CUDA 13 (falls back to CPU) |
| 2026-11-08 | `results/m9/{lmo-tok4x,ycbv-tok4x2}/eval_det_model_*` | **LM-O test / YCB-V test**, BOP default detections | M9 bullet 2: TensorRT 10.14 engines of `tok4x` / `tok4x2` (`last.pt`), profile 1/8/16, A100, official BOP19 eval | LM-O: fp16 **AR 0.6685** (VSD 0.4911, MSSD 0.6415, MSPD 0.8729), fp32 0.6683, PyTorch 0.6685; YCB-V: fp16 **AR 0.7017**, PyTorch 0.7021 | fp16 loses ≤ 0.0004 AR (exit < 0.005). Max output diff fp16 2.0e-2 / fp32 7.6e-6 |
| 2026-11-08 | `results/m9/latency` | One LM-O test image (8 instances, repeated for 16), A100, `dev` 16 CPUs | M9 bullet 3: latency vs N, PyTorch fp32 / amp vs TensorRT fp32 / fp16 (`tok4x`), network alone and full per-image pipeline | TensorRT fp16 pipeline **5.3 / 9.3 / 14.2 / 24.0 ms** (N = 1 / 4 / 8 / 16), network 1.8 / 3.5 / 5.7 / 11.0; PyTorch amp pipeline 10.8 / 13.5 / 17.0 / 29.5 | Pose stage 70 FPS at 8 objects; with the default detector (82 ms) ~10 FPS. GPU normalisation + CPU R, t recovery: pipeline 44 → 14 ms at N = 8 |
| 2026-11-08 | `outputs/FastPose-ViT/runs/<ablation run>/<variant>/eval_det`, `results/paper/ablation_ar_raw.md` | **LM-O test**, default detections | Paper §2: all 24 ablation models (M3–M7, `last.pt`) on official BOP19 AR (`scripts/paper_ablation_eval.sh`; M3 driller-only with a driller targets file) | driller: site 0.844 / 0.845, legacy **0.785**, decoupled 0.844, domain_gap 0.841; 8 objects: std 0.629 / 0.645, dg 0.641 / 0.637, 2× 0.650 / 0.650, dg4x 0.661, heads 2× 0.650 / 0.650, token 0.659 / 0.651, both 0.660 / 0.661, sym 0.655 / 0.655, bobj 0.646 / 0.650, bmix 0.650 / 0.654, **tok4x / tok4x2 0.668 / 0.664** | One metric for the paper's ablation tables. Clear effects: site depth (+0.06), compute; no measurable effect: domain_gap, symmetric loss, crop source, decoupled loss (earlier val/driller ADD overstated augmentation and conditioning). Table in `docs/BOP.md` §2 |
| 2026-11-08 | `results/paper/{lmo,ycbv}_tok4x*_{modelpth,lastpt_retimed}` | **LM-O / YCB-V test**, default detections | Paper §2: `model.pth` (best val) vs `last.pt`, and `last.pt` re-timed with the current `estimate_image` | LM-O 0.6686 vs **0.6685**, YCB-V 0.7021 vs **0.7021**; BOP time 134 → **98.5 ms** (LM-O), 118 → **93.6 ms** (YCB-V) incl. detector, AR unchanged | Checkpoint choice doesn't matter; keep `last.pt` |
| 2026-11-08 | `results/paper/{per_object_ar_*,model_size,det_map_default_*,det_iou_default_*}` | LM-O / YCB-V test | Paper §2: per-object official AR (`scripts/bop_per_object_ar.py`, recomputed AR = official), model size / FLOPs, default-detection quality (`scripts/detection_map.py`) | LM-O per object 0.357 (eggbox) – 0.826 (can); YCB-V 0.386 (wood block) – ~0.9; 86.1 M params, 55.5 GMACs per crop at 384 px (17.6 at 224); default detections mAP 0.657 (LM-O) / 0.758 (YCB-V), 6.3 % wrong target boxes on LM-O (eggbox 27 %) | Tables in `docs/BOP.md` §2.1–2.3 |
| 2026-11-09 | `outputs/FastPose-ViT/runs/lmo-m9-det{tiny,tiny2,small}-001/<variant>/{det_test,eval_det}`, `results/m9/det-small-trt/` | **LM-O test**, own LW-DETR detections | M9 stretch: multi-class LW-DETR (tiny ×2 seeds, small), 50 epochs; small exported to TensorRT fp16 (gather sampler instead of GridSample) | mAP tiny 0.628 / 0.631, small 0.656 (TensorRT 0.657); pose AR (`tok4x`) **0.645 / 0.650 / 0.654**, full TensorRT pipeline 0.654 | Default detections 0.668; detector 4.2 ms mean (3.3 median) per image |
| 2026-11-09 | `outputs/FastPose-ViT/runs/lmo-m10-r224tok4x{,2}-001` | **LM-O test**, default detections | M10: `tok4x` recipe at 224 px (`vit_b_16`), 2 seeds | AR **0.643 / 0.643** vs 384 px 0.668 / 0.664 | Pipeline 7.1 vs 14.1 ms at N = 8 |
| 2026-11-11 | `outputs/FastPose-ViT/runs/lmo-m10-{absrot,scratch,bbox0,bbox20,legacy}{,2}-001`, `results/paper/section3/` | **LM-O test**, default detections | Paper §3 ablations: symtok2x baseline (0.655 / 0.655) with one change, 2 seeds | absrot 0.546 / 0.548, scratch 0.516 / 0.517, legacy 0.610 / 0.625, bbox0 0.625 / 0.625, bbox20 0.650 / 0.651 | Every kept choice confirmed; per-object tables in `section3/` |
| 2026-11-11 | `results/m9/det-small-trt/latency_{384,224}` | One LM-O test image, A100 | Latency re-run with our detector's time | TensorRT pipeline 5.3 / 9.4 / 14.1 / 24.1 ms (384) and 4.3 / 5.8 / 7.1 / 10.6 ms (224) for N = 1/4/8/16; + detector 4.2 ms → **55 FPS** at N = 8 (88 FPS at 224 px) | Detector time added, same GPU |
| 2026-11-12 | `outputs/FastPose-ViT/runs/lmo-m10-augnone{,2}-002`, `results/paper/section3/per_object_augnone*.md` | **LM-O test**, default detections | Augmentation study: no pixel augmentation (baseline `domain_gap` 0.655 / 0.655) | AR **0.626 / 0.629** (−0.028) | Glue −0.14, can −0.05; other aug variants running |
| 2026-11-14 | `outputs/FastPose-ViT/runs/lmo-m10-aug{color,blur,noise,simple,strong}{,2}-002` | **LM-O test**, default detections | Augmentation study, baseline symtok2x (`domain_gap` p = 0.5: 0.655 / 0.655) | color 0.649 / 0.641, blur 0.628 / 0.630, noise 0.627 / 0.627, color + blur + noise 0.641 / 0.643, `domain_gap` p = 0.8 0.641 / 0.631 | Keep `domain_gap` p = 0.5 |
| 2026-11-14 | `outputs/FastPose-ViT/runs/lmo-m10-bbox{152,302}-002` | **LM-O test**, default detections | Box jitter 15 / 30 %, seed b | 0.651 / 0.643 | `bbox{5,52,15,30}-002` failed (node gpu028, 7 pods "ContainerStatusUnknown"), relaunched as `-003` |
| 2026-11-14 | `results/m11/ensemble/{lmo,ycbv}` | **LM-O / YCB-V test**, default detections | M11 B4: pose average of the two seed models (`scripts/pose_ensemble.py`) | LM-O 0.6689, YCB-V 0.7048 | Seeds 0.668 / 0.664 and 0.694 / 0.702: not worth 2× pose time |
| 2026-11-14 | `results/m11/rgbd/{lmo,ycbv}` | **LM-O / YCB-V test**, default detections + test depth | M11 C1: RGB-D variant, depth correction of the RGB model's poses (`scripts/depth_correction.py`) | LM-O **0.750** (VSD 0.594, MSSD 0.785, MSPD 0.871), YCB-V **0.854** (0.802 / 0.912 / 0.849); 0.18 / 0.16 s per image | RGB 0.668 / 0.702; PBR-only RGB-D entries: LM-O 0.69–0.80, YCB-V 0.76–0.91 |
| 2026-11-14 | `results/m11/rgbd_splat/{lmo_tok4x,lmo_tok4x2,lmo_ownsmall_trt,ycbv_tok4x,ycbv_tok4x2}` | **LM-O / YCB-V test** + test depth | M11 C2–C3: fast GPU depth correction (`--method splat`), all RGB results | LM-O 0.749 / 0.752, own detector + TensorRT 0.738 (24 ms per image); YCB-V 0.849 / 0.851; correction 3–4 ms per image | Same AR as the renderer version (0.750) at ~85 ms |
| 2026-11-15 | `outputs/FastPose-ViT/runs/lmo-m10-bbox{5,52,15,30}-003`, `bbox{152,302}-002` | **LM-O test**, default detections; LM real val with noisy boxes | Box jitter 5 / 15 / 30 % per side (default 10 %: 0.655 / 0.655, val 0.8725 / 0.8746) | Test AR seed a / b: 5 % 0.647 / 0.654, 15 % 0.656 / 0.651, 30 % 0.646 / 0.643; validation: 5 % 0.858 / 0.858, 15 % 0.869 / 0.872, 30 % 0.851 / 0.849 | 10 % kept (15 % within noise, 5 % and 30 % worse) |
| 2026-11-15 | `outputs/FastPose-ViT/runs/lmo-m11-{ctx,ctx125,ctx15,occl,tlw}{,2}-001` | **LM-O test**, default detections; LM real val noisy boxes (`eval_valn`, own score) | M11 phase A, 2× baseline `symtok2x` + one change, 2 seeds a / b | Test AR: ctx 0.623 / 0.621, ctx125 0.624 / 0.627, ctx15 0.623 / 0.619, occl 0.657 / 0.651, tlw 0.659 / 0.658. Validation: ctx 0.815 / 0.810, ctx125 0.815 / 0.817, ctx15 0.813 / 0.816, occl 0.875 / 0.878, tlw 0.871 / 0.876 | Context variants −0.03 on both; occl and tlw within noise of baseline (0.655, 0.874). GT-box validation (`eval_val`): LM real val 0.89–0.91, PBR val 0.889 (ctx) to 0.907 (tlw); in `$RUN/<variant>/eval_val{,n}/val_score.txt` |
| 2026-11-15 | `outputs/FastPose-ViT/runs/lmo-m12-base{,2}-001` | **LM-O test**, default detections; LM real val noisy boxes | Audit W1: the ablation baseline re-trained on the current code (2×, token, domain_gap, symmetric loss, 10 % jitter) | AR 0.650 / 0.654 (VSD 0.470, MSSD 0.617, MSPD 0.868 on average); validation score 0.875 / 0.871; GT-box validation LM real 0.893 / 0.892, PBR 0.904 / 0.901 | Original `symtok2x` 0.655 / 0.655, 0.874: the older code changed nothing beyond the seed spread |
| 2026-11-15 | `results/lmo-m12-single-s1/{eval_det,eval_test}` | **LM-O test**, default detections / GT boxes | Audit W4: second seed of the 8 single-object models (`lmo-m12-obj{1,5,6,8,9,10,11,12}s1-001`), CSVs merged | AR **0.666** (VSD 0.488, MSSD 0.633, MSPD 0.877), GT boxes 0.682 | Seed 0: 0.672 / 0.688; mean 0.669 vs shared 0.666: same |
| 2026-11-15 | `outputs/FastPose-ViT/runs/tless-m10-tok4x-002`, `tless-m10-tok4x2-002` | **T-LESS test_primesense**, default detections / GT boxes | M10 T-LESS: one shared model, 30 objects, `tok4x` recipe (4×, 633,120 samples per epoch), 2 seeds | Detections: AR **0.7616 / 0.7624** (VSD 0.650 / 0.651, MSSD 0.701 / 0.701, MSPD 0.933 / 0.935); GT boxes 0.8127 / 0.8120 | Detector costs 0.05 AR; per-object tables in the run folders and `docs/BOP.md` §1.4; leaderboard comparison not done |
| 2026-11-15 | `results/m11/rgbd_splat/tless_tok4x{,2}` | **T-LESS test** + test depth | M11 C6: RGB-D variant (`--method splat`) of the two T-LESS RGB results | **0.858 / 0.859** (VSD 0.789 / 0.790, MSSD 0.853 / 0.854, MSPD 0.931 / 0.933), 0.101 s per image | RGB 0.762 / 0.762; median shift −11 / −12 mm, 3.2 ms per image |
| 2026-11-16 | `outputs/FastPose-ViT/runs/ycbv-m10-heads2x{,2}-001` | **YCB-V test**, default detections (GT boxes in brackets); PBR val | Per-object heads instead of the class token, 2×, 2 seeds (compare `tok2x{,2}`) | AR 0.674 / 0.678 (GT boxes 0.679 / 0.681) against token 0.681 / 0.689 (0.687 / 0.701); best PBR-val score 0.811 / 0.812 against 0.834 / 0.832 | Token better on YCB-V too: +0.009 AR (mean 0.685 vs 0.676), +0.02 on validation; same direction as LM-O (+0.005). Paper: supplementary conditioning row + one sentence |
| 2026-11-16 | `outputs/FastPose-ViT/runs/lmo-m12-detsmall2-001`, `ycbv-m10-detsmall-001`; `results/m11/rgbd_splat/ycbv_ownsmall` | **LM-O / YCB-V test**, own LW-DETR small detections (`scripts/m9_det.sh small`) | LM-O seed b of the detector (pose model `tok4x`); YCB-V detector, 50 epochs, pose model `tok4x2`; YCB-V RGB-D (C7) | LM-O AR 0.656 (VSD 0.487, MSSD 0.631, MSPD 0.849), mAP 0.653; YCB-V AR **0.681** (VSD 0.574, MSSD 0.663, MSPD 0.805), mAP 0.702 (default 0.758), RGB-D **0.827** (0.773 / 0.885 / 0.822) | LM-O: same as seed a (0.654). YCB-V: own detector −0.021 AR (−0.024 with depth), a larger loss than LM-O (−0.014); mAP lower than the default one, AR100 the same. Paper: supplementary detection and RGB-D tables, one sentence in §4.4 |
| 2026-11-16 | `outputs/FastPose-ViT/runs/lmo-m11-tlwocc{,2}-001` | **LM-O test**, default detections; LM real val noisy boxes (`eval_valn`) | M11 A6, 2× baseline + occlusion + translation weight ×2, 2 seeds | Test AR 0.664 / 0.658 (VSD 0.486 / 0.478, MSSD 0.634 / 0.626, MSPD 0.871 / 0.869); `eval_valn` 0.885 / 0.880; `eval_val` LM real 0.902 / 0.896, PBR 0.909 / 0.907 | Baseline 0.652 / 0.873: +0.009 on both, above the seed spread (0.004), both seeds above both baseline seeds. Combination chosen after seeing the single results. Paper: supplementary `tab:supp:m11` |
| 2026-11-17 | `outputs/FastPose-ViT/runs/lmo-m13-tlwocc4x-001` | **LM-O test**, default detections; LM real val noisy boxes | M11 A7: occlusion + translation weight ×2 at 4×, 1 seed | AR 0.669 (VSD 0.490, MSSD 0.642, MSPD 0.875); `eval_valn` 0.887; `eval_val` LM real 0.901, PBR 0.926 | `tok4x` / `tok4x2`: 0.668 / 0.664, 0.886 / 0.880: +0.003 / +0.004, not confirmed; the 2× gain does not carry to 4×. Not adopted |
| 2026-11-18 | `outputs/FastPose-ViT/runs/lmo-m13-tlwocc4x2-001` | **LM-O test**, default detections; LM real val noisy boxes | M11 A8: occlusion + translation weight ×2 at 4×, seed b | AR 0.6666 (VSD 0.487, MSSD 0.640, MSPD 0.873); `eval_valn` 0.8855; `eval_val` LM real 0.900, PBR 0.923 | `tok4x2` 0.664 / 0.880. 2-seed means at 4×: test 0.668 vs 0.666, validation 0.886 vs 0.883: **adopted under the new rule (2026-11-18)** |
| 2026-11-19 | `outputs/FastPose-ViT/runs/lmo-m13-base8x-001` (seed a; `base/`) | LM-O test (detections) + LM real / PBR validation | M11 B2: baseline recipe (no `tlw` + `occl`) at 8× (SPE 8 × 42,208), 150 epochs, ViT-B 384 px | Test AR **0.6728** (VSD 0.495, MSSD 0.647, MSPD 0.876); real validation score GT boxes 0.8968, noisy boxes 0.8861; PBR validation 0.9292 (training best 0.9242 on the 2000-image subset) | vs `tok4x` / `tok4x2` (4×): test 0.6685 / 0.6639, noisy 0.8862 / 0.8796, GT 0.8969 / 0.8909, PBR 0.9217 / 0.9216. One seed only: seed b `lmo-m13-base8x2-001` launched 2026-11-20. The high 0.92 is the PBR validation, not real images. |
| 2026-11-20 | M15 `lmo-m15-vitb512-002`, `lmo-m15-vitb5122-001` | LM-O | ViT-B/16 at 512 px, tlwocc recipe, 2× | noisy validation 0.8809 / 0.8839 (mean 0.8824), test AR 0.6572 / 0.6638 (0.6605) | baseline `tlwocc{,2}` 0.8847 / 0.8802 (0.8825), 0.6636 / 0.6577 (0.6606): tied (−0.0001 on both), not adopted |
| 2026-11-20 | M15 `lmo-m15-vitl5122-001` (seed b only) | LM-O | ViT-L/16 at 512 px, 2 GPUs | noisy validation 0.8902, GT-box validation 0.9001, PBR validation 0.9417, test AR 0.6814 | vs the `tlwocc` means 0.8825 / 0.6606: +0.008 / +0.021. Seed a (`lmo-m15-vitl512-002`) is still running: the rule needs both |
| 2026-11-21 | M15 `lmo-m15-vitl512-002` (seed a) | LM-O | ViT-L/16 at 512 px, 2× | noisy validation 0.8916, test AR 0.6899 | With seed b (0.8902 / 0.6814): means **0.8909 / 0.6857** vs `tlwocc` 2× 0.8825 / 0.6606 (+0.008 / +0.025): **adopted at 2× under the rule**; the 4× runs `lmo-m15x4-vitl512{,2}-001` decide the final size |
| 2026-11-21 | M19 `lmo-m19-cnxa-001` (seed a, own refiner, lr 5e-5), iterations 0 / 1 / 2 / 3 / 4 | LM-O | ConvNeXt-B refiner on `tlwocc4x` seed a poses | noisy validation 0.8873 / **0.8978** / 0.8950 / 0.8913 / 0.8882; test AR 0.6692 / **0.6977** / 0.6970 / 0.6928 / 0.6897 | **+0.0105 validation, +0.0285 test with 1 iteration**. `cnxb` (lr 1e-4): 0.8891 / 0.6700 at 1 iteration, no gain. Seed b `cnxa2` and `cnxs{,2}` still running: adopt with the 2-seed mean |
| 2026-11-21 | M17 `tless-m17-{base2x,real2x}{,2}-001` | T-LESS | 2×, PBR only vs PBR + real | noisy validation 0.8709 / 0.8721 vs 0.8743 / 0.8817; test AR 0.7635 / 0.7622 vs 0.7749 / 0.7785 | Real images: +0.0065 validation, +0.0139 test (means 0.8715 / 0.7629 vs 0.8780 / 0.7767); separate leaderboard categories |
| 2026-11-22 | M18 `lmo-m18-densesymx5{,2}-001` (symmetric XYZ targets + XYZ weight 5) | LM-O | multi-head net, 2×; baselines `densesym` PnP 0.9063 / 0.7100, `densex5` PnP 0.9122 / 0.7165, `tlwocc` 2× 0.8825 / 0.6606 | PnP val .9137 / .9105, test .7180 / .7128; `pnp_init` .9140 / .9105, .7182 / .7133; `dr_rot_pnp_t` .9118 / .9093, .7121 / .7084; direct .8928 / .8887, .6911 / .6821 | **PnP means 0.9121 / 0.7154: adopted over `densesym` (+0.006 / +0.005); tied with `densex5` (−0.000 / −0.001).** Recipe for the long runs: symmetric targets + weight 5 + PnP (all three PnP modes agree within 0.003, no weak mode on symmetric objects). Total over `tlwocc` 2×: +0.030 validation, +0.055 test |
| 2026-11-22 | M19 `lmo-m19-{cnxp,cnxq}{,2}-001` (refiner trained on small / PnP-like noise), iterations 0..4 on the PnP poses of `lmo-m18-densesym-001` (0 = .9055 / .7065) | LM-O | PnP poses, 2× | `cnxp` a / b val at 1 / 4 iterations: .9058 / .9066, .9061 / .9075, test .7063 / .7069, .7067 / .7080; `cnxq`: val .9042 / .9065 at 1, then falls to .8995 / .9014 at 4; test .7092 / .7083 at 1 | **No refinement gain on PnP poses** (best 2-seed mean: `cnxp`, 4 iterations, +0.0016 val / +0.0010 test; `cnxq` +0.000 val / +0.002 test at 1 iteration): not adopted. PnP alone (0.9055 / 0.7065) beats direct poses plus the refiner (0.9049 / 0.6980), and the two gains do not stack with this refiner |
| 2026-11-21 | M19 on dense poses: `scripts/m19_on_dense.sh` (refiner `cnxs` seed a on the PnP poses of `lmo-m18-densesym-001`), iterations 0 / 1 / 2 | LM-O | PnP poses, 2× | noisy validation .9055 / **.9076** / .9044; test AR .7065 / **.7091** / .7039 | The refiner (trained on direct-regression errors) adds only +0.002 validation / +0.003 test to PnP poses, against +0.018 / +0.029 on direct poses: PnP poses are already about as accurate. A refiner trained on the PnP error distribution is the next candidate (not launched) |
| 2026-11-21 | M15 `lmo-m15-vitb640{,2}-002` | LM-O | ViT-B/16 at 640 px, 2× | noisy validation 0.8787 / 0.8806, test AR 0.6683 / 0.6653 | Means 0.8797 / 0.6668 vs `tlwocc` 2× 0.8825 / 0.6606: validation −0.003, test +0.006: **not adopted at 2×** (validation worse). 4× runs `lmo-m15x4-vitb640{,2}-001` still running |
| 2026-11-21 | M18 `lmo-m18-densesym{,2}-001` (symmetric XYZ targets), `densexl2-001` | LM-O | multi-head net, 2×, weight 1; baseline `tlwocc` 2× 0.8825 / 0.6606 | densesym a / b: PnP val .9055 / .9070, test .7065 / .7135; `pnp_init` .9061 / .9070, .7072 / .7145; `dr_rot_pnp_t` .9069 / .9061, .7049 / .7098; direct .8899 / .8903, .6865 / .6867. densexl2: PnP .8953, .6923 | **Symmetric XYZ targets adopted** (PnP means val 0.9063 / test 0.7100 vs `dense` 0.9048 / 0.7033; direct 0.8901 / 0.6866 vs 0.8863 / 0.6809): they fix `pnp_init` and `dr_rot_pnp_t`, which were bad on symmetric objects (now all three PnP modes within 0.001 val). Weight 0.2 (`densexl`) worse (means PnP 0.8964 / 0.6935). Next: symmetric targets with weight 5 (`densesymx5`) |
| 2026-11-21 | M19 `lmo-m19-{cnxa,cnxb,cnxs}{,2}-001` (own refiner, seeds a / b), iterations 0 / 1 / 2 / 3 / 4 | LM-O | on `tlwocc4x` seed a poses; baseline 0.8873 / 0.6692 | `cnxa`: val .8978 / .8997, test .6977 / .6969; `cnxb`: .8891 / .8887, .6700 / .6696; **`cnxs` (noise `dr`): val .9055 / .9044, test .6989 / .6972** at 1 iteration; 2 iterations about equal, 3–4 worse | **Adopted: `cnxs`, 1 iteration (2-seed means val 0.9049, test 0.6980: +0.018 / +0.029).** Refiner trained on the DR error distribution wins; lr 1e-4 (`cnxb`) is useless. Cost ~+30 ms per iteration |
| 2026-11-21 | M18 `lmo-m18-{dense,densex5,densexl}{,2}-001`, seeds a / b (densexl2, densesym{,2} still running) | LM-O | multi-head net, 2×, XYZ weight 1 / 5 / 0.2; baseline `tlwocc` 2× val 0.8825, test 0.6606 | PnP on mask + XYZ: **dense val .9051 / .9045, test .7049 / .7017; densex5 val .9100 / .9144, test .7164 / .7166**; densexl .8974, .6947 (seed a). Direct regression with the dense heads: dense .8846 / .8880, .6799 / .6819; densex5 .8898 / .8938, .6910 / .6885. PnP from the regressed pose: .8811, .6633 (dense), regressed rotation + PnP translation: .8545, .6155 (dense) | **PnP on dense output adopted** (means: val 0.9048 / test 0.7033 for weight 1, +0.022 / +0.043 over `tlwocc` 2×; weight 5 best so far: 0.9122 / 0.7165). The dense heads alone also help direct regression (+0.006 / +0.020). `pnp_init` and `dr_rot_pnp_t` are not better than direct. Weight 0.2 worse |
| 2026-11-20 | M16 `{ycbv,tless}-m16-obj*{,s1}-001`, pooled with `scripts/m16_adopt.sh` | YCB-V, T-LESS | one network per object, 2 seeds | YCB-V: noisy validation 0.6814 / 0.6664 (12 objects in the held-out videos), test AR (detections) 0.7180 / 0.7069. T-LESS: 0.8457 / 0.8437, test 0.7376 / 0.7342 | baselines `tok4x{,2}`: YCB-V 0.6930 / 0.6885, test 0.6936 / 0.7021; T-LESS 0.8768 / 0.8793, test 0.7616 / 0.7624. Not adopted (YCB-V test +0.015 but validation −0.017; T-LESS −0.033 / −0.026) |
| 2026-11-20 | M17 `lmo-m17-real2x-002`, `lmo-m17-real2x2-001` | LM-O, PBR + real (LM `test_all`, no BOP19 images) | tlwocc recipe, 2× | noisy validation 0.9706 / 0.9724, test AR 0.6756 / 0.6858 (mean 0.6807) | vs `tlwocc{,2}` 0.8825, 0.6606: test +0.020; the validation gain is inflated (the real validation views are ~5° from real training views). PBR + real category |
| 2026-11-20 | M17 / M11 B3 `ycbv-m14-real2x-001`, `ycbv-m14-real2x2-002` | YCB-V, PBR + real (`train_real`) | 2× | noisy validation on 4 held-out real videos 0.7418 / 0.7613 (after `summary_remean.py` on seed a), test AR (detections) 0.7570 / 0.7711 (0.7641) | vs `tok2x{,2}`: 0.6684 / 0.6829 (0.6757), 0.6814 / 0.6885 (0.6849): +0.076 / +0.079, adopted for the PBR + real category |
| 2026-11-20 | M19 R1, `$NAS_HOME/results/m19_r1_{lmo,ycbv,tless}/` | LM-O, YCB-V, T-LESS test | CosyPose PBR-only refiner (happypose, Panda3D) on our poses, 1–4 iterations | test AR 0 / 1 / 2 / 3 / 4 iterations: LM-O 0.669 / 0.616 / 0.584 / 0.567 / 0.557; YCB-V 0.694 / 0.693 / 0.694 / 0.689 / 0.689; T-LESS 0.762 / 0.743 / 0.731 / 0.722 / 0.719 | no gain; sanity check with noisy ground truth in the M19 R1 bullet |
(Paths relative to `$NAS_HOME`.)

---

## 10. Next session: start here

0. **Tracker (updated 2026-11-18 19:30 CET = 18:30 UTC; times below are Swiss time, the cluster logs are UTC). Scope changed on 2026-11-18 (§0, D9–D13): accuracy first, report due 2027-01-01, decision gate 2026-12-04.**

   | Task | Status |
   |---|---|
   | **Running jobs** | |
   | M11 B2 8× budget, LM-O: seed a `lmo-m13-base8x-001` **finished 2026-11-19 08:44 UTC (exit 0)**; seed b `lmo-m13-base8x2-001` launched 2026-11-20 (~68 h) | Seed a (4× baseline `tok4x{,2}` in brackets): test AR 0.6728 (0.6685 / 0.6639), real validation noisy boxes 0.8861 (0.8862 / 0.8796), GT boxes 0.8968 (0.8969 / 0.8909), PBR validation 0.9292 (0.9217 / 0.9216). Test +0.007 over the 2-seed mean, validation noisy tied with seed a. **Decide with the 2-seed mean when seed b is done** (also against `tlwocc4x{,2}`: test 0.6692 / 0.6666, noisy validation 0.8873 / 0.8855) |
   | M17 / M11 B3 YCB-V PBR + real: `ycbv-m14-real2x-001`, `ycbv-m14-real2x2-001` (2×) | Preempted, Pending (epochs 123 and 91 of 150); resume from `last.pt`. When done: `scripts/summary_remean.py` on `eval_valreal{,n}`, then the rule against `tok2x{,2}` |
   | **Launched 2026-11-18 ~19:45 CET (all Pending at first: the cluster was full, 29 nodes × 8 GPUs in use)** | |
   | M15 `lmo-m15-{vitb512,vitb640}{,2}-001` (1 GPU), `lmo-m15-vitl512{,2}-001` (2 GPUs), `lmo-m15-vith518{,2}-001` (4 GPUs, `NUM_WORKERS=11`) | Queued. Estimates once running: B/512 ~19 h, B/640 ~29 h, L/512 ~32 h, H/518 ~2.5–3 days. Baseline `tlwocc{,2}` 2× (test 0.661, `eval_valn` 0.882) |
   | M17 `lmo-m17-real2x{,2}-001`; `tless-m17-{base2x,real2x}{,2}-001` | Queued (~11 h LM-O, ~42 h T-LESS). LM-O baseline `tlwocc{,2}`; T-LESS baseline `base2x{,2}` |
   | M16 wave 1: `ycbv-m16-obj{1..21}-001` (seed 0) | Queued (~3 h each) |
   | M16 rest (user 2026-11-19: submit now, not held for M15): `tless-m16-obj{1..30}-001`, `ycbv-m16-obj{1..21}s1-001`, `tless-m16-obj{1..30}s1-001` | Submitted 2026-11-19 ~01:00 CET (81 jobs, ~3 h each) |
   | Jobs removed by the cluster (status "Deleted", no exit code, not stopped by us; most likely preemption with ~120 of our jobs queued over the project share, but such jobs were requeued before): `lmo-m17-real2x-001` (after 40 s, 2026-11-18 23:36 UTC), `lmo-m15-vitb640{,2}-001` (in epoch 2, ~23:55 UTC). Resubmitted as `lmo-m17-real2x-002`, `lmo-m15-vitb640{,2}-002` (from scratch: only 1 epoch was lost). Also `ycbv-m14-real2x2-001` (2026-11-19 ~00:30 UTC, in epoch 102): resubmitted as **`ycbv-m14-real2x2-002`, resuming from the epoch-100 `last.pt`** (the old `real2x2/` folder is copied into the new run folder first; the old folder is kept). It runs the current code, so its `eval_valreal{,n}` need no re-meaning (`real2x-001` still does). **When checking jobs, look for "Deleted" and resubmit with a new name** | Queued |
   | M16 baselines: T-LESS `tok4x` / `tok4x2` on the new `val_real` (`DS=tless scripts/m14_valreal_eval.sh`, dev) | Done 2026-11-18: `eval_valrealn` **0.877 / 0.879 (mean 0.878)**, GT boxes 0.910 / 0.911 |
   | M15 fix 2026-11-19: `lmo-m15-vitb512-001` stopped with "unknown variant" (a trailing "2" was read as seed b: `vitb512` → `vitb51`); parsing fixed (`8c894c0`), relaunched as `lmo-m15-vitb512-002`; `lmo-m15-vitl512-001` (would have failed the same way) replaced by `lmo-m15-vitl512-002`. The seed-b jobs `vitb5122`, `vitl5122` parse correctly | Queued |
   | **Decisions waiting for the user (2026-11-20 night)** | None open. Done on request: our own refiner (M19 R2, jobs `lmo-m19-*`), `densesym` launched (`lmo-m18-densesym{,2}-001`, queued) |
   | **Still to do** | |
   | M18 multi-head (DR + mask + XYZ, PnP at test): `lmo-m18-{dense,densex5,densexl}{,2}-001` (XYZ weight 1 / 5 / 0.2; 2 seeds) | **Launched 2026-11-20 ~17:20 UTC**, **2026-11-20 20:15 UTC: `dense`, `dense2`, `densex5`, `densexl` running, `densex52`, `densexl2` Pending.** Symmetric-XYZ variant `densesym` coded (M18.6a), not launched. 2× budget, 1 GPU each, ~10 h once running; baseline `tlwocc{,2}`. When done: the four pose modes are evaluated by the script itself (`eval_val_<mode>`, `eval_valn_<mode>`, `eval_det_<mode>`); compare by the adoption rule. Seed b of `densex5` is `densex52`, of `densexl` is `densexl2` |
   | M19 R1 CosyPose PBR refiner on our poses (separate venv) | To do |
   | M16 LM-O result (`eval_valn` of the 16 per-object models) | Done 2026-11-18: validation 0.879 vs 0.883, test 0.669 vs 0.666 → not adopted on LM-O |
   | **Report (M20)** | |
   | Decision gate: final recipe per dataset | Fri 2026-12-04 |
   | Final models (3 datasets × PBR only / PBR + real × 2 seeds) + RGB-D | W8–W9 |
   | EPFL report on `paper/` (~20 pages; Pablo Sarró Sánchez, Mathieu Salzmann) | W9–W11, due 2027-01-01 |
   | T-LESS leaderboard block; tables in both categories (PBR only, PBR + real) | With the finals |
   | First compile on Overleaf (`main.tex`, `supplementary.tex`) | To do (user) |
   | **Closed by the 2026-11-18 meeting** | |
   | Positioning vs PViT-6D / PoseDETR | Decided: accuracy first (D9) |
   | Venue, page limit, authors | Report: no limit, ~20 pages, 2 authors; publication later with Pierre Ancey (D12) |
   | Jetson Orin latency | Dropped for the semester project (limitation) |
   | Release plan | Optional: GitHub + Hugging Face (D13) |
   | M11 A8 `lmo-m13-tlwocc4x2-001` | Done 2026-11-18: AR 0.6666, validation 0.8855; **`tlw` + `occl` adopted** (2-seed means +0.002 test, +0.003 validation) |
   | **Done 2026-11-15** | |
   | bbox 5 / 15 / 30 % (paper recipe table, supplementary S3, `docs/BOP.md`); T-LESS RGB, RGB-D (C6), per-object tables; M11 phase A (+ validation, re-trained baseline `base`, W1); single-object second seed (W4); paper / README / BOP.md / TASKS.md updated | Done |
   | **Done 2026-11-14** | |
   | Augmentation study (keep `domain_gap` p = 0.5) | Done |
   | Supplementary split (`paper/supplementary.tex`), learning-rate fix in the draft | Done |
   | M11 B4 seed ensemble (not kept) | Done |
   | M11 C1–C5 RGB-D variant (LM-O 0.749 / 0.752, YCB-V 0.849 / 0.851, 24 ms pipeline) | Done |
1. Start Claude inside `dev`, from the repo root (§4).
2. Work on branch `multi-object` (created 2026-10-20). The §5 report to Pierre is deferred until M8.
3. ~~Set up `.venv` + `bop_toolkit`~~ (done 2026-10-20, see §7 M0). Install with `.venv/bin/pip`, not uv.
4. ~~Tiny `train` job that reads BOP files~~ (done 2026-10-20, `lmo-m0-access-004`). New jobs normally show in `runai list` within ~10 s; a one-off ~9 min delay happened once (see §7 M0). ~~`.gitignore`/pytest~~ (done). ~~Golden SPEED eval with `.venv`~~ (done, byte-identical). ~~bop_toolkit GT-as-prediction check~~ (done, AR = 1.0). **M0 complete.**
5. ~~Image 1.60 with `libegl1`~~ (done 2026-10-21: `dev` and `train` run 1.60; AR 1.0 and golden 0.0093 without the EGL workaround, see M3 first bullet).
6. **M1** started 2026-10-20 (while the image 1.60 build is running): ~~`tests/test_rotation_convention.py`~~ (done, 9 tests). ~~`tests/test_translation_param.py`~~ (done, 13 tests; the `src/utils.py` part of the refactor is included). ~~Refactor~~ (done: `num_objects`, `unpack_batch`/`forward_model`, per-sample K in `CoordinateTransformer`). ~~Checkpoint/resume~~ (done; kill/resume verified). **M1 complete.**
7. **History of M2–M9 (full running notes moved to `docs/TASKS_HISTORY.md` on 2026-11-14, W10; the same facts are in §1, §7 and §9).** In short: M2 BOP loader and labels verified (reprojection, round trip); M3 single-object baseline B1 (driller) and the `site` depth, `domain_gap` ablations; M4 shared 8-object model (class token, 4× budget, final `tok4x`); M5 symmetric loss (kept, no measurable gain); M6 official BOP19 eval with GT boxes (LM-O AR 0.689) and the training speed-ups; M7 default detections (LM-O AR 0.668) and the crop-source ablation; M8 YCB-V (AR 0.702) and the symmetry offsets; M9 ONNX / TensorRT and latency.
- Way of working (decided 2026-10-20): go through §7 one bullet at a time. Do the step, update this file, then stop until the user has checked it.
- Experiments (decided 2026-10-22): GPUs are effectively unlimited. For open choices, run the candidate configs as parallel train jobs (plus baseline repeats for noise) instead of assuming one; see `AGENTS.md`. Add any newly found task as a §7 bullet.
- Leftovers the user planned to delete by hand (safe to delete, nothing uses them): `$NAS_HOME/results/bop-m0-gtpred/eval_egl/`, `$NAS_HOME/results/bop-m0-gtpred/eval/tmp1792521949/`, and the superseded 224 px chain runs `outputs/FastPose-ViT/runs/lmo-m3-abl-001/`, `-002/` (never delete run folders without the user).
- After every bullet: commit on `multi-object` with a 2–4 word message with every word starting in a Capital Letter (e.g. `BOP Access Verified`), then `git push origin multi-object` to the fork `PabloSarro/FastPose-ViT-MO` (never to `upstream`), and don't show `claude` as the author of the pushes but myself (`PabloSarro`).
