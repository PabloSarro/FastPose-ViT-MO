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
| M0 Setup and data access | W1 | `[x]` | Golden SPEED eval reproduced (0.0093) and bop_toolkit gives AR ≈ 1.0 on GT-as-prediction | Golden SPEED 0.0093 reproduced (byte-identical); bop_toolkit GT-as-prediction AR = **1.0** (VSD via EGL workaround, `dev` and `train`). Image 1.60 with `libegl1` done 2026-10-21 (M3 first bullet; AR 1.0 and golden 0.0093 without the workaround) |
| M1 Refactor and safety net | W2 | `[x]` | SPEED golden diff = 0, pytest green, smoke train OK | Golden byte-identical; 57 tests; smoke val 0.1317; kill/resume continues exactly (LR, losses); done 2026-10-20, a week early |
| M2 BOP data loader (LM-O) | W3–4 | `[x]` | Reprojection and label round-trip tests pass, throughput OK | Started 2026-10-21. Index done: train 320,959 rows (LM PBR 0–47), test 1,445 (= BOP19 targets). `BOPDataset.__getitem__`, reprojection and label round-trip tests and visual check done. Loader 730 samples/s = 2.8× GPU (257). **Closed 2026-10-21**; the optional roll augmentation is deferred |
| M3 Single-object baseline B1 | W4–5 | `[~]` | B1 metrics logged in §9 | Started 2026-10-21. Image 1.60 done; `train.py --dataset LMO` works; overfit check passed (256 driller instances: 0.43 mm, 0.017°) |
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

## 4. Agent setup

- **Decided 2026-10-20: `AGENTS.md` is the only agent-instructions file; no `CLAUDE.md` is needed.**
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
  - **Finding: `domain_gap` is far too strong for LM.** Eleven transforms at p = 0.5 each (incl. solarize, sun flare, tone curve, gray) made most crops in a preview grid unrecognizable. Hence `standard` as default; compare both in the M3 ablations (new decision D5 in §8).
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
- **Exit:** official LM-O AR with oracle boxes, with in-loop MSSD/MSPD within ±0.01 of the official values. Multi-GPU: N = 1 identical to before, and a measured 1 vs N speed-up table in §9.

### M7 — Real detections (W9–11)
- [ ] Download the BOP default detections for seen objects: [`bop23_default_detections_for_task1.zip`](https://bop.felk.cvut.cz/media/data/bop_datasets_extra/bop23_default_detections_for_task1.zip) (GDRNPPDet).
- [ ] Measure whether detector boxes match `bbox_visib` or `bbox_obj` better (mean IoU), and adapt the crop source and jitter.
- [ ] Re-evaluate LM-O with detections; keep the top `inst_count` per object by detection score.
- [ ] (Stretch) Multi-class LW-DETR so the Jetson pipeline is self-contained. Today `object_detector/` is hard-coded to one class.
- **Exit:** first BOP-comparable LM-O AR, compared with §2.2.

### M8 — YCB-V (W11–12)
- [ ] Send Pierre the data issues from §5 before starting YCB-V. *Moved here from M0 (2026-10-20): LM-O isn't affected by them.*
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

**GPU budget (A100, one job at a time):** about 50–65 GPU-h for the core path, 85–100 with stretch goals. Long runs are chained with resume across 12 h jobs. Multi-GPU (DDP) is planned in M6; before that every training run uses 1 GPU.

---

## 8. Open decisions and questions

| ID | Question | Decide at |
|---|---|---|
| D1 | Detections: BOP defaults only, or also our own multi-class LW-DETR? | M7 |
| D2 | Object conditioning: per-object heads vs class tokens (vs both) | M4 |
| D3 | Crop source for training: `bbox_visib` vs `bbox_obj` | M2 / M7 |
| D4 | `visib_fract` threshold for training instances (default 0.1) | M2 |
| D5 | BOP pixel augmentation: `standard` (default) vs `domain_gap` (too strong in the preview); black bands vs `crop_context` | M3 |
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
| 2026-10-20 | `outputs/FastPose-ViT/runs/lmo-m0-access-004` | LM PBR / LM-O models | `scripts/check_bop_access.py` | `BOP ACCESS OK (lmo)` | BOP readable from `train` jobs |
| 2026-10-20 | `results/eval-m0-venv-golden` | SPEED test | same as golden, `.venv/bin/python` | score **0.0093** | `results.json` byte-identical to golden |
| 2026-10-20 | `results/bop-m0-gtpred-wrapper` | LM-O test (BOP19 targets) | GT poses as predictions, `scripts/bop_eval.sh` | AR **1.0** (VSD 1.0, MSSD 1.0, MSPD 1.0) | Eval pipeline OK; 53 s in `dev` |
| 2026-10-20 | `results/eval-m1-refactor` | SPEED test | golden command after the M1 refactor | score **0.0093** | `results.json` byte-identical to golden |
| 2026-10-20 | `results/smoke-m1-refactor` | SPEED val | `smoke-02` command, 2 epochs, `dev` | val score 0.1317 | vs 0.1400 (smoke-02); within noise |
| 2026-10-20 | `results/smoke-m1-resume` | SPEED val | `smoke-02` command, 3 epochs, SIGKILL after epoch 1 + `--resume auto` | val score 0.2137 / 0.1403 / **0.1005** | Resume continues LR and losses; 3-ep uninterrupted ref 0.1021 |
| 2026-10-21 | `results/eval-m3-image160` + `lmo-m3-image160-001` | LM-O test (GT as prediction) + SPEED val | Image 1.60 checks: BOP eval in `dev` and in a train job, golden eval | AR **1.0** (dev and train); SPEED **0.0093** byte-identical | Native libEGL, no workaround; 57 tests pass |
| 2026-10-21 | `results/lmo-m3-overfit` | 256 fixed LM PBR driller instances (train = val) | Overfit check, best SPEED config, batch 32, 150 epochs, no aug | translation 0.43 mm, rotation 0.017° (train loss 0.0015) | Pipeline learns its labels end to end; 10.5 min in dev |

(Paths relative to `$NAS_HOME`.)

---

## 10. Next session: start here

1. Start Claude inside `dev`, from the repo root (§4).
2. Work on branch `multi-object` (created 2026-10-20). The §5 report to Pierre is deferred until M8.
3. ~~Set up `.venv` + `bop_toolkit`~~ (done 2026-10-20, see §7 M0). Install with `.venv/bin/pip`, not uv.
4. ~~Tiny `train` job that reads BOP files~~ (done 2026-10-20, `lmo-m0-access-004`). New jobs normally show in `runai list` within ~10 s; a one-off ~9 min delay happened once (see §7 M0). ~~`.gitignore`/pytest~~ (done). ~~Golden SPEED eval with `.venv`~~ (done, byte-identical). ~~bop_toolkit GT-as-prediction check~~ (done, AR = 1.0). **M0 complete.**
5. ~~Image 1.60 with `libegl1`~~ (done 2026-10-21: `dev` and `train` run 1.60; AR 1.0 and golden 0.0093 without the EGL workaround, see M3 first bullet).
6. **M1** started 2026-10-20 (while the image 1.60 build is running): ~~`tests/test_rotation_convention.py`~~ (done, 9 tests). ~~`tests/test_translation_param.py`~~ (done, 13 tests; the `src/utils.py` part of the refactor is included). ~~Refactor~~ (done: `num_objects`, `unpack_batch`/`forward_model`, per-sample K in `CoordinateTransformer`). ~~Checkpoint/resume~~ (done; kill/resume verified). **M1 complete.**
7. **M2** started 2026-10-21: ~~`src/bop_dataset.py` index~~ (done; train 320,959 rows, test 1,445). ~~`__getitem__`~~ (done: `BOPDataset`, single-warp crop, site/allocentric labels, `standard` pixel aug by default). ~~Reprojection test~~ (done; found and fixed BOP's +1 px box convention, see M2). Visualisations in `vis/` (see `vis/README.md`; rerun `bash vis/run_all.sh` after data/label changes). ~~Label round trip~~ (done: ≤ 1e-6 on PBR, 7e-6 on LM-O test after snapping GT R to the nearest rotation). ~~Visual check~~ (done: 64-crop grids for train, test and val_real look right). ~~Loader throughput~~ (done: loader 730 samples/s vs GPU 257 at batch 32 with compile; no staging; use batch 32). **M2 closed 2026-10-21** (optional roll augmentation deferred). **M3** started: ~~image 1.60~~, ~~overfit 256 driller instances~~ (passed: 0.43 mm / 0.017°; `train.py --dataset LMO` works). **Running since 2026-10-21 11:14: job `lmo-m3-b1-001`** (M3 single-object training, driller), run folder `$NAS_HOME/outputs/FastPose-ViT/runs/lmo-m3-b1-001/`. 150 epochs (ends at a warm-restart cycle end; 200 would stop mid-cycle at a high LR), ~6.8 h expected. Train 42,214 PBR driller crops (1,319 batches of 32), val all 1,988 driller `val` crops (PBR 48–49 + LM real), standard pixel aug, 10 % jitter, `site` depth, best SPEED config, `--resume auto`. Launched without monitoring at the user's request. Next session: check `runai list` and `log.txt`, `bash rcp/stop.sh lmo-m3-b1-001` once finished, plot with `vis/training_curves.py`, then the M3 evaluation bullet (LM real val and LM-O test for driller). Use `--resume auto` in every training command from now on.
- Way of working (decided 2026-10-20): go through §7 one bullet at a time. Do the step, update this file, then stop until the user has checked it.
- After every bullet: commit on `multi-object` with a 2–4 word message with every word starting in a Capital Letter (e.g. `BOP Access Verified`), then `git push origin multi-object` to the fork `PabloSarro/FastPose-ViT-MO` (never to `upstream`), and don't show `claude` as the author of the pushes but myself (`PabloSarro`).
