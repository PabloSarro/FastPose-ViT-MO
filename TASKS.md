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
| Must keep | RGB only (no depth) · direct regression (no PnP, no render-and-compare refinement) · real-time / Jetson deployable · SPEED keeps working (score 0.0093) |
| Datasets | **LM-O first** (train on LM `train_pbr`), then YCB-V, then T-LESS (stretch) |
| Timeline | Semester project, ~14 weeks |

Week calendar (Mondays): W1 Oct 19 · W2 Oct 26 · W3 Nov 2 · W4 Nov 9 · W5 Nov 16 · W6 Nov 23 · W7 Nov 30 · W8 Dec 7 ·
W9 Dec 14 · W10 Dec 21 · W11 Dec 28 · W12 Jan 4 · W13 Jan 11 · W14 Jan 18. W10–W11 are holidays, a good slot for long runs.

---

## 1. Status board

| Milestone | Weeks | Status | Exit criterion | Result |
|---|---|---|---|---|
| M0 Setup and data access | W1 | `[~]` | Golden SPEED eval reproduced (0.0093) and bop_toolkit gives AR ≈ 1.0 on GT-as-prediction | |
| M1 Refactor and safety net | W2 | `[ ]` | SPEED golden diff = 0, pytest green, smoke train OK | |
| M2 BOP data loader (LM-O) | W3–4 | `[ ]` | Reprojection and label round-trip tests pass, throughput OK | |
| M3 Single-object baseline B1 | W4–5 | `[ ]` | B1 metrics logged in §9 | |
| M4 Shared 8-object model | W6–7 | `[ ]` | ≥ 90 % of B1 on B1's object; no object at 0 ADD | |
| M5 Symmetries | W7–8 | `[ ]` | Eggbox/glue improve, others unchanged | |
| M6 Official BOP eval (GT boxes) | W8–9 | `[ ]` | Official LM-O AR (oracle) | |
| M7 Real detections | W9–11 | `[ ]` | First BOP-comparable LM-O AR | |
| M8 YCB-V | W11–12 | `[ ]` | YCB-V AR + per-object table | |
| M9 Jetson / deployment | W12–13 | `[ ]` | Latency vs number of objects | |
| M10 Write-up (+ T-LESS stretch) | W13–14 | `[ ]` | Report and final tables | |

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
| `tests/` | Empty (`__init__.py` only) | Geometry, loader, loss and SPEED-golden tests |
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

## 4. Agent setup: CLAUDE.md vs AGENTS.md

- **They serve the same purpose:** project instructions loaded at session start. Codex reads `AGENTS.md`. Claude Code (≥ 2.1.277; installed 2.1.304) also reads `AGENTS.md`, **but only when no `CLAUDE.md`, `.claude/CLAUDE.md` or `CLAUDE.local.md` exists in the working directory or above it** ([docs](https://code.claude.com/docs/en/memory)).
- **Decision: keep `AGENTS.md`, don't rename it, don't create `CLAUDE.md`.**
  - `rcp-init` refreshes the "Running on the RCP cluster" section of `AGENTS.md`, and Codex would lose a renamed file.
  - Project-specific notes go in the `## Project notes` section at the end of `AGENTS.md`, as the RCP tutorial recommends.
  - This file (`TASKS.md`) is the detailed tracker. It is referenced from AGENTS.md, not imported, to keep the context small.
- If you ever need a `CLAUDE.md` or `CLAUDE.local.md` (the latter is already in `.gitignore`), its **first line must be `@AGENTS.md`**. Otherwise Claude silently stops reading AGENTS.md and loses the RCP rules.
- **Start Claude from the repo root inside the `dev` job:** `dev`, then `cd $NAS_HOME/FastPose-ViT && claude`. The 2026-10-20 session was started from `$NAS_HOME` on what looked like the jumphost (no GPU, `RCP_JOB` unset). There, AGENTS.md is only loaded on demand and the `.claude/settings.json` permission rules don't apply.
- **Session routine:** read §1 and §10 → work → update §1, §9, §10 → update AGENTS.md project notes if a convention changed.

---

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

**Report to Pierre (dataset owner):** deferred on 2026-10-20. None of these issues affects LM-O, so send them before starting YCB-V (M8).
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
9. **Risks.**
   - VSD needs a headless renderer (EGL or OSMesa in the container). It is tested in M0.
   - NAS I/O: each PBR image is decoded about 7 times per pass.
   - PBR→real domain gap.
   - LR scaling at large batch.
   - Test leakage if LM-O test is used for any decision.

---

## 7. Milestones and sub-milestones

### M0 — Setup and data access (W1)
- [x] Repair `$NAS_HOME/BOP_Datasets` symlink (done 2026-10-20; `ls $NAS_HOME/BOP_Datasets/lmo` works).
- [ ] Send Pierre the data issues from §5. *Deferred (2026-10-20): LM-O comes first and isn't affected. Send before M8 (YCB-V).*
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
- [ ] Verify BOP read access **inside a `train` job**, not just `dev`, with a tiny job that reads one PBR jpg, `scene_gt.json` and one `models_eval` ply. Commit `1c08f08` added the supplemental groups for this.
- [ ] Add `.gitignore` exceptions (`!tests/**`, and any config JSON you add); check `pytest tests/` runs.
  - Found on 2026-10-20: `6b3c430` already removed the `test*` rule (so `tests/test_*.py` is no longer hidden), as well as the rules for `.pytest_cache`, `results`, `runs`, `checkpoints`, `temp` and `.ruff_cache`. It also added `AGENTS.md`, which has no effect because the file is tracked. `*.json` is still ignored.
  - The removals are intentional: these folders are empty for now. Add ignore rules back (e.g. `.pytest_cache`) once they start filling up.
- [ ] Golden SPEED eval with `.venv/bin/python` (command in `docs/RCP_WORKFLOW.md` §5). It must print score 0.0093; diff its `results.json` against `$NAS_HOME/results/eval-rcp-check/results.json`.
- [ ] bop_toolkit sanity check:
  - Build `scripts/make_bop_tree.sh` (symlinked tree in `$NAS_HOME/datasets/bop/`).
  - Write a **GT-as-prediction** CSV for LM-O and run `eval_bop19_pose.py`. Expect AR ≈ 1.0.
  - This tests the VSD renderer. If EGL fails, try `NVIDIA_DRIVER_CAPABILITIES=all`, or OSMesa in a new image tag.
- **Exit:** golden SPEED 0.0093 reproduced, and toolkit AR ≈ 1.0.

### M1 — Safety net and refactor, no behaviour change (W2)
- [ ] `tests/test_rotation_convention.py`:
  - `quat2dcm(q)ᵀ == quaternion_to_rotation_matrix(q)` on random q.
  - ego → allo → ego round trip.
  - `R_rel(t)·ẑ = t̂`.
  - Roll equivariance.
- [ ] `tests/test_translation_param.py`:
  - The new code path with `K = Camera.K` and 1920×1200 matches the old one bit for bit.
  - The `site` mode forward/inverse round trip works with the LM and YCB-V cameras.
- [ ] Refactor:
  - Optional `K` / `img_wh` / `depth_mode` / `diameter` in `src/utils.py` and `CoordinateTransformer`.
  - `unpack_batch` in `src/engine.py`.
  - `num_objects` in `FastPoseViT` (default 1).
- [ ] Checkpoint/resume in `src/train.py`:
  - Atomic `last.pt` in `$RUN_DIR` containing model, optimizer (Muon), scheduler/LR index, GradScaler, epoch, global iteration, best metric and RNG states.
  - `--resume auto`.
  - Test: kill a 3-epoch smoke run after epoch 1, resume, and check that the curve continues.
- **Exit:** golden diff = 0 (≤ 1e-6), pytest green, 2-epoch SPEED smoke run behaves as before.

### M2 — BOP data loader for LM-O (W3–4)
- [ ] `src/bop_dataset.py` index: one row per visible instance (`visib_fract ≥ 0.1`, valid bbox), stored as numpy arrays and cached as `.npz` under `$NAS_HOME/cache/`. Expect ≈ 343k train rows and 1,445 LM-O test targets.
- [ ] `__getitem__`:
  - Crop from `bbox_visib` with jitter, as a single `warpAffine` to a square crop with zero padding.
  - Labels: allocentric R and `U` in `site` mode; also return K, `obj_idx`, diameter and ids.
  - Pixel augs without OpticalDistortion.
- [ ] Reprojection test: project `models_eval` points with the GT pose. The projected bbox must match `bbox_obj` with IoU ≥ 0.9, and using `Rᵀ` must clearly fail. Run it on 30 PBR and 30 LM-O instances.
- [ ] Label round trip: dataset labels → inverse functions (K, bbox) → GT ego pose, within 1e-4.
- [ ] Visual check: a 64-crop grid with projected axes/points saved as PNG, including eggbox and glue.
- [ ] Loader throughput with 16 workers. If it is below about 2× GPU speed, stage the PBR RGB (~5.4 GB) to local disk or group crops per image.
- [ ] (Optional) exact roll augmentation, off by default, tested by checking projected points stay inside the warped mask.
- **Exit:** all tests pass and the overlays look right.

### M3 — Single-object baseline B1: original architecture, GT boxes (W4–5)
- [ ] Overfit 256 instances of one object to near-zero loss in about 10 min. This catches label bugs.
- [ ] Train FastPose-ViT (ViT-B/16-384, best SPEED config except pixel aug on, `site` depth) on **one object** (8 driller or 5 can) from LM PBR.
- [ ] Evaluate on the LM real val split and on LM-O test for that object: rotation error (°), translation error (mm), ADD@0.1d, MSSD/MSPD.
- [ ] Short ablations at 224: legacy vs `site` depth; relative vs decoupled translation loss.
- **Exit:** B1 logged in §9. This is "the existing architecture on BOP".

### M4 — Shared multi-object model (W6–7)
- [ ] Per-object heads (option a) with object-balanced sampling, trained on all 8 LM-O objects.
- [ ] Per-object table (ADD(-S), MSSD/MSPD recall) on val and, at the end of the milestone only, on LM-O test.
- [ ] (Ablation) per-object class tokens (option b), and a combination of a and b.
- **Exit:** one network reaches ≥ 90 % of B1 on B1's object, and no object is at 0 ADD.

### M5 — Symmetries (W7–8)
- [ ] `SymmetricRotation6DLoss` with a padded symmetry table. Unit tests:
  - Loss = 0 for every `R_gt·S_k`.
  - Equal to the plain loss for non-symmetric objects.
  - Finite gradients.
- [ ] In-loop validation metrics on the GPU (ADD-S, MSSD, MSPD), and model selection on `(AR_MSSD+AR_MSPD)/2`.
- **Exit:** eggbox (10) and glue (11) MSSD/MSPD improve, and the other objects stay within noise.

### M6 — Official BOP evaluation with GT boxes (oracle) (W8–9)
- [ ] `src/predict_bop.py` writes BOP19 CSV:
  - File name `fastposevitmo_lmo-test.csv` (no `_` inside the method name).
  - Columns `scene_id,im_id,obj_id,score,R,t,time`; R row-major, **t in mm**.
- [ ] Run `eval_bop19_pose.py` and report AR_VSD, AR_MSSD, AR_MSPD and AR.
- **Exit:** official LM-O AR with oracle boxes, with in-loop MSSD/MSPD within ±0.01 of the official values.

### M7 — Real detections (W9–11)
- [ ] Download the BOP default detections for seen objects: [`bop23_default_detections_for_task1.zip`](https://bop.felk.cvut.cz/media/data/bop_datasets_extra/bop23_default_detections_for_task1.zip) (GDRNPPDet).
- [ ] Measure whether detector boxes match `bbox_visib` or `bbox_obj` better (mean IoU), and adapt the crop source and jitter.
- [ ] Re-evaluate LM-O with detections; keep the top `inst_count` per object by detection score.
- [ ] (Stretch) Multi-class LW-DETR so the Jetson pipeline is self-contained. Today `object_detector/` is hard-coded to one class.
- **Exit:** first BOP-comparable LM-O AR, compared with §2.2.

### M8 — YCB-V (W11–12)
- [ ] Symlink tree + index. Train on PBR only unless Pierre fixes `train_real`. There are two cameras, so per-sample K matters.
- [ ] Continuous symmetries (objects 13, 18) and discrete ones (1, 16, 19, 20, 21).
- **Exit:** YCB-V AR plus a per-object table.

### M9 — Jetson / deployment (W12–13)
- [ ] ONNX export with dynamic batch (`dynamic_shapes`). ONNX Runtime vs PyTorch max abs diff < 1e-3 (fp32).
- [ ] TensorRT fp16 engine with an optimization profile for 1/8/16 crops. AR change < 0.005.
- [ ] Latency per image for N = 1, 4, 8, 16 objects (A100, and Jetson Orin Nano if available). The SPEED engine must still build.
- **Exit:** latency table with the real-time claim quantified.

### M10 — Write-up (W13–14), T-LESS as a stretch
- [ ] T-LESS only if M8 is done by W12.
- [ ] Final tables:
  - AR on LM-O and YCB-V vs RGB-only BOP entries without refinement.
  - Ablations (depth mode, conditioning, symmetry loss, augmentation, resolution 224/384).
  - Latency.
- [ ] README "BOP" section, `scripts/bop_*.sh`, merge `multi-object` into `main`.

**GPU budget (A100, one job at a time):** about 50–65 GPU-h for the core path, 85–100 with stretch goals. Long runs are chained with resume across 12 h jobs.

---

## 8. Open decisions and questions

| ID | Question | Decide at |
|---|---|---|
| D1 | Detections: BOP defaults only, or also our own multi-class LW-DETR? | M7 |
| D2 | Object conditioning: per-object heads vs class tokens (vs both) | M4 |
| D3 | Crop source for training: `bbox_visib` vs `bbox_obj` | M2 / M7 |
| D4 | `visib_fract` threshold for training instances (default 0.1) | M2 |
| D5 | Backbone resolution for real time: 224 vs 384 | M3 / M9 |
| D6 | Several instances of the same object (T-LESS) | M10 |

**For the supervisor:**
- [ ] Are IPD / XYZ-IBD / HB, downloaded on 2026-10-19, meant for this project?
- [ ] GPU-hour budget.
- [ ] Access to a Jetson Orin Nano.
- [ ] Expected deliverable (report, paper, presentation).

---

## 9. Results log

| Date | Run / folder | Dataset | Config | Metrics | Notes |
|---|---|---|---|---|---|
| 2026-09-30 | `results/eval-pretrained-vit-b-16-384` | SPEED test | HF `vit_b_16_384.pth`, best config | E_T 0.0221 m, E_R 0.0073 rad, score **0.0093** | Matches paper |
| 2026-09-30 | `results/smoke-train` | SPEED val | 2 epochs, README config | val score 0.1379 | First smoke run |
| 2026-10-13 | `results/eval-rcp-check` | SPEED test | same as above, RCP `dev` | score **0.0093** | Golden file |
| 2026-10-14 | `outputs/FastPose-ViT/runs/smoke-02` | SPEED val | 2 epochs, `train` job | val score 0.1400 | RCP `train` works |
| 2026-10-14 | `outputs/FastPose-ViT/runs/speed-short-001` | SPEED val | 3 epochs | val score 0.1021 | ~1.8 min/epoch |

(Paths relative to `$NAS_HOME`.)

---

## 10. Next session: start here

1. Start Claude inside `dev`, from the repo root (§4).
2. Work on branch `multi-object` (created 2026-10-20). The §5 report to Pierre is deferred until M8.
3. ~~Set up `.venv` + `bop_toolkit`~~ (done 2026-10-20, see §7 M0). Install with `.venv/bin/pip`, not uv.
4. Next M0 bullet: a tiny `train` job that reads BOP files (access check). Then `.gitignore`/pytest, the golden SPEED eval with `.venv/bin/python`, and the GT-as-prediction bop_toolkit check.
- Way of working (decided 2026-10-20): go through §7 one bullet at a time. Do the step, update this file, then stop until the user has checked it.
- After every bullet: commit on `multi-object` with a 2–4 word ALL-CAPS message (e.g. `BOP ACCESS VERIFIED`), then `git push origin multi-object` to the fork `PabloSarro/FastPose-ViT-MO` (never to `upstream`).
