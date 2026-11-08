# Paper features and guidelines for the follow-up paper

Purpose: a model (or person) who writes the follow-up paper on the multi-object BOP work should
read this file first. Part A describes the published FastPose-ViT paper (structure, style,
conventions, what it already claims). Part B lists **every source to read** for the work done since
then and how to turn it into a paper with the same features.

Source: `docs/WACV - FastPose-ViT Original Paper.md` (text extracted from the CVF Open Access PDF;
the extraction merged words and scrambled the two-column layout and the tables, so read numbers
from the tables carefully, and check them against the PDF if in doubt). Reference: P. Ancey,
A. Price, S. Javed, M. Salzmann (EPFL, Swiss Data Science Center), *FastPose-ViT: A Vision
Transformer for Real-Time Spacecraft Pose Estimation*, WACV 2026, pp. 7873–7882,
[arXiv 2512.09792](https://arxiv.org/abs/2512.09792).

---

## Part A. The original paper

### A.1 Format
- CVF/WACV two-column conference format: 8 pages of body (pp. 7873–7880) + 2 pages of references
  (38 numbered entries, `[n]` style, sorted alphabetically by first author). A supplementary
  material is referenced for extra backbones and the spatial-augmentation derivation.
- 4 figures, 11 tables, 17 numbered equations.

### A.2 Structure (section by section)

| Section | Content | Length / role |
|---|---|---|
| Title | "*FastPose-ViT: A Vision Transformer for Real-Time Spacecraft Pose Estimation*": method name, colon, what + domain + the selling point (real time) | |
| Figure 1 (top of p. 1, teaser) | Pipeline overview: detector → crop → (1) ViT regresses intermediate parameters, (2) geometric recovery to full-frame pose, (3) export, quantization, Jetson deployment. Caption numbers the steps | Sets up the whole paper in one picture |
| Abstract (1 paragraph, ~200 words) | Problem (6DoF from one image, critical for in-orbit servicing / debris removal) → limitation of SOTA (iterative PnP, slow, ill-suited to edge) → proposal (ViT, direct regression, crops, mathematical formalism, "apparent rotation") → result (beats non-PnP, competitive with PnP on SPEED) → deployment number (Jetson Orin Nano, 75 ms sequential, up to 33 FPS concurrent) | |
| 1. Introduction | Context paragraph → hybrid vs direct methods (direct = efficient but less accurate) → "To address this limitation, we propose…" → **3 bulleted contributions** (architecture; geometric reformulation of targets with closed-form correction; PyTorch→TensorRT pipeline with FP16 without accuracy loss) → datasets and headline claim ("sets a new state-of-the-art for non-PnP-based methods… competitive with top-tier PnP… real-time on a small edge device") | ~0.75 column |
| 2. Related Work | Three run-in bold paragraphs: *Spacecraft Pose Estimation* (datasets, hybrid PnP top performers with their latency, direct methods URSONet / E-PoseNet), *Vision Transformers for Pose Estimation* (PViT-6D on BOP: complex, no weights, hard to reproduce; FoundationPose: multi-stage, not for edge), *Rotation Representations* (6D continuous representation of Zhou et al.; "our key innovation lies in reformulating the regression target itself") | Each paragraph ends with how we differ |
| 3. Methodology | Goal sentence (predict T ∈ R³, R ∈ SO(3) from one RGB image) → **3.1** ViT architecture (crop from detector, classification MLP discarded, [CLS] → single linear projection; deeper MLPs tried, no gain; Table 1 backbones: ViT-B-224/16 85.8 M / 17.5 GFLOPs, ViT-B-384/16 86.1 M / 55.5 GFLOPs) → **3.2** data augmentations (pixel: Albumentations, p = 0.5 each; bounding box: ±10 % per side, hand-annotated boxes + LW-DETR; spatial: in-plane rotation with pose relabeling Eq. 1–2, ranges [−20°, 20°] or [160°, 200°]) → **3.3** geometric reformulation: 3.3.1 translation via normalized coordinates (U_z crop-depth proxy with scale factors s_x = W/(αw), s_y = H/h, α = 1.6, least squares Eq. 3–4; lateral offsets U_x, U_y from the bbox centre Eq. 5–7), 3.3.2 apparent rotation R′ = ΔR·R (axis u = T̂ × e_z, angle θ = arccos(T̂ᵀe_z), Rodrigues Eq. 8–10; 6D first two columns + Gram–Schmidt Eq. 11–12) | ~3 columns; every formula is derived step by step from the pinhole model |
| 4. Experiments and Results | **4.1** Setup with run-in heads: *Data*, *Evaluation Metrics* (official ESA E_T, E_R, Eq. 13–14), *Implementation Details* (SWAG weights, Muon without weight decay, cosine warm restarts T0 = 10, Tmult = 2, 310 / 150 epochs, LR 1e-4 ↔ 1e-6, batch 8; detector settings), *Training Losses* (relative Frobenius on R′ first two columns Eq. 15, L2 on U Eq. 16, sum Eq. 17) → **4.2** SOTA comparison (Table 2 SPEED; Tables 3–4 SPEED+ lightbox / sunlamp; an honest explanation of why we lag on SPEED+, in 2 numbered factors) → **4.3** ablations "on SPEED… unless otherwise specified, all experiments use our final proposed configuration on the ViT-B-224/16 model": 4.3.1 transfer learning (Table 5), 4.3.2 geometric formulation and cropping (Table 6, 2×2: full/cropped × absolute/relative rotation), 4.3.3 augmentations (Tables 7–9, incl. bbox perturbation 0/10/20/30 %) → **4.4** model compression and deployment (Jetson Orin Nano 25 W, CUDA 12.6, TensorRT 10.7, Python 3.10; 5 pipeline stages listed; Table 10 ViT-only vs end-to-end per backbone and precision; Table 11 shared overheads; sequential vs non-blocking throughput; FP16 helps the detector 2× and ViT-B-384, not ViT-B-224) | ~3.5 columns |
| 5. Conclusion | One paragraph: restate the method and the "apparent rotation"; the key contribution ("for the first time, real-time ViT-based pose estimation on embedded edge hardware, up to 33 FPS"); what the ablations validate; limitations (domain gap, bbox sensitivity) and the way forward (training data closer to real imagery) | ~0.3 column |

Figures: (1) teaser pipeline, (2) concept figure for the apparent rotation, (3) qualitative
best / average / worst predictions on both datasets with GT axes solid and predictions dashed,
errors under each image as "T [m] | R [deg]", (4) failure case: bad vs good bounding box.

### A.3 Writing style
- **Voice**: first-person plural ("we propose", "we find", "we attribute"), active, present tense
  for the method, past tense for what was done ("we hand-annotated", "we experimented").
- **Claim pattern**: problem → limitation of existing work → our idea → quantified result. Claims
  are scoped: "state of the art *for non-PnP methods*", "competitive *with* PnP-based techniques",
  "*on* the SPEED dataset". Superlatives only with a qualifier ("for the first time … on
  embedded edge hardware").
- **Honesty about weaknesses**: a whole paragraph explains the SPEED+ gap (competition teams
  tuned augmentations with hidden-test knowledge, rebuilt the 3D model) and turns it into a
  strength (reproducibility, released data only, generic augmentations). Limitations are restated
  in the conclusion.
- **Motivating with numbers**: competitor latency and power (150 ms on an RTX 2080 Ti at 250 W vs a
  1–25 W spacecraft budget).
- **Terminology** defined once and then reused verbatim: "direct regression", "hybrid",
  "PnP-based", "apparent rotation / apparent orientation", "crop-depth proxy", "normalized
  translation targets", "end-to-end pipeline". Model names are written `ViT-B-384/16 (Ours)`.
- **Math**: every symbol is defined right after its equation; derivations go from the pinhole model
  to the closed-form inverse used at inference ("we can invert this equation to recover…").
  Vectors / matrices: T, R, K, U = (U_x, U_y, U_z), R′, ΔR, [u]×, e_z.
- **Paragraph-level run-in headings** in bold (Data., Evaluation Metrics., Implementation
  Details., Training Losses., Pixel-Level Augmentations., …) instead of deep subsections.
- **Justify each design choice** in one sentence, often with an ablation pointer ("we found no
  consistent improvement", "as shown in Table 7").
- Short practical asides that show engineering care (box clipping and discarding degenerate boxes,
  rotation ranges chosen so the object stays visible, FP16 slower than FP32 for the small model
  because of conversion overheads).

### A.4 Table and number conventions
- Comparison tables: one row per method, a **"PnP?" column (Yes / No / ?)**, metric columns with
  units and arrows (`E_T [m]↓`, `E_R [deg]↓`), our rows marked "(Ours)", sorted by the main
  metric, "?" = information not released (stated in the caption).
- Ablation tables: the final configuration appears as one row of each ablation (e.g. "Spatial Augs
  Only 0.0231 / 0.4469", "10 %"), so the reader sees the chosen setting in context.
- Precision: 4 decimals for SPEED errors, 3 for SPEED+; latency as mean ± std in ms.
- Captions: a full sentence saying what is compared and on which data; "?" explained.
- Hardware and software versions stated for every latency number.

### A.5 What the original paper already claims (do not re-claim as new)
ViT-B/16 direct regression from crops with a single linear head; SWAG pre-training; the
normalized translation targets (U_x, U_y, U_z with α); the apparent rotation with a closed-form
correction; 6D rotation + Gram–Schmidt; relative Frobenius / L2 losses; Muon without weight decay;
cosine warm restarts; bbox perturbation 10 %; in-plane rotation augmentation; pixel and style
augmentations; a one-class LW-DETR detector; TensorRT FP16 on a Jetson Orin Nano. The PViT-6D
discussion already mentions BOP: the follow-up must position itself against it.

---

## Part B. Guidelines for the follow-up paper (multi-object, BOP)

### B.1 Read everything since the WACV paper, in this order
1. `AGENTS.md` (`## Project notes`): goal, constraints (RGB only, direct regression, no PnP or
   refinement, Jetson real time, SPEED must keep working), conventions (BOP translations in mm,
   `cam_R_m2c` = SPEED rotation convention, image-rotation augmentation invalid for BOP).
2. `TASKS.md`, all of it, in this order: §0 goal and scope → §1 status board (one line per
   milestone with the key numbers) → §2 baselines (SPEED reference; BOP leaderboard tables) → §5
   data inventory and known issues → §6 design notes (verified in the code) → **§7 milestones
   M0–M10, every bullet** (the full story: decisions, failed ideas, prerequisites found, tables) →
   §8 open decisions (D1–D8) → **§9 results log** (one row per measured result, with its run
   folder) → §10 next steps.
3. `docs/BOP.md`: the draft of the final tables (LM-O, YCB-V, ablations, deployment, limitations,
   reproduce). Rows marked *pending* must be filled from §9 before writing.
4. Git history of the work: `git log --reverse --stat main..multi-object` (98+ commits, one per §7
   bullet, Title Case messages). Read the diffs of the commits that introduce a method change (e.g.
   `git log --oneline main..multi-object` then `git show <hash>`).
5. Code, for exact definitions (cite the code, not memory): `src/bop_dataset.py` (index, crops,
   BOP +1 px box convention, `pixel_transform`, augmentations), `src/utils.py` (translation
   parametrisations: `legacy` = the paper's α-based U_z, `site` = the new one; apparent rotation),
   `src/engine.py` (`CoordinateTransformer`, per-sample K), `src/models.py` (object conditioning:
   heads / class token / both), `src/losses.py` (symmetric rotation loss with translation offsets),
   `src/train.py` (object-balanced sampler, `--bop_*` flags, model selection), `src/bop_metrics.py`
   and `src/bop_evaluate.py` (metrics, detections, BOP CSV, timing, TensorRT backend),
   `optimization/bop_export.py`, `bop_trt.py`, `bop_latency.py` (deployment),
   `object_detector/bop_detector.py` (multi-class LW-DETR), `scripts/m*_*.sh` (the exact command of
   every experiment).
6. Run folders and result files referenced in §9 (paths relative to `$NAS_HOME`):
   `outputs/FastPose-ViT/runs/<job>/` (`info.txt` = exact command and git commit, `log.txt`,
   `<variant>/eval*/table_*.md`, `scores_bop19.json`), `results/m9/` (ONNX/TensorRT checks,
   `latency/latency.md`).
7. `vis/` and `vis/figures/` (`bash vis/run_all.sh`): crops, reprojection checks, training curves
   (`vis/training_curves.py RUN_DIR`); reuse or extend them for the paper figures.
8. This file's Part A and the original paper itself, for style.

### B.2 Rules for facts and numbers
- Every number in the paper must come from §9 / a run folder / `docs/BOP.md`, with the
  configuration that produced it. Quote **official BOP19 AR** (`scripts/bop_eval.sh`) for
  comparisons; own MSSD / MSPD recalls and ADD are for ablations only and must be named as such.
- Say which boxes were used: BOP default detections (comparable to the leaderboard) vs GT boxes
  (oracle, upper bound) vs our own detector.
- Reported models evaluate `last.pt` (final epoch), not `model.pth`.
- Leaderboard numbers in `TASKS.md` §2 / `docs/BOP.md` were read with an automatic extractor:
  verify each quoted entry on the BOP website, and whether it uses refinement (the "?" rows).
- Seed noise: ablation configs were run with 2 seeds ("a / b"); differences below the seed spread
  (~0.005–0.015) are "no measurable effect", not gains.
- Latency: A100 only (no Jetson measured for BOP). Detector time of the default detections is the
  one reported by its authors, not measured by us. Do not mix the WACV Jetson numbers with the BOP
  A100 numbers without saying so.
- Results still running when this file was written (2026-11-08): own detector (`lmo-m9-det*`),
  224 px ablation (`lmo-m10-r224*`). Check §1 / §9 for their final values.

### B.3 Proposed outline (same skeleton as the WACV paper)
- **Title**: same pattern, e.g. "*FastPose-ViT-MO: Real-Time Multi-Object 6D Pose Estimation by
  Direct Regression*" (method, colon, task, selling point).
- **Teaser figure**: one image → detector → N crops batched through one shared ViT with per-object
  class tokens → N poses; latency vs N inset.
- **Abstract**: limitation of BOP RGB SOTA (correspondences + PnP, refinement, one model per
  object, slow) → our proposal (one direct-regression network for all objects, PBR only) → AR on
  LM-O / YCB-V and position among PBR-only RGB methods → latency (TensorRT fp16, 14 ms for 8
  objects on an A100; end-to-end with detector).
- **1 Introduction**: 3–4 bulleted contributions, candidates (choose from §7 what holds up):
  (i) multi-object extension with a per-object class token (vs per-object heads / per-object
  models, M4, M10); (ii) camera- and object-size-independent translation target (`site` depth) and
  per-sample intrinsics, needed across datasets (M1, M3); (iii) symmetry-aware rotation loss with
  translation offsets for objects whose symmetry axis is off the origin (M5, M8); (iv) a deployment
  study: batched TensorRT fp16 with no AR loss and latency vs number of objects, plus the pipeline
  fixes (GPU normalisation, CPU pose recovery, M9) and our own multi-class detector (M9 stretch).
- **2 Related Work**: BOP RGB methods (correspondence + PnP: ZebraPose, GDRNPP, EPOS, CDPN;
  render-and-compare: CosyPose, SurfEmb, MegaPose; direct / transformer: PViT-6D, PoseDETR),
  multi-object vs per-object models, symmetries, fast detectors (LW-DETR).
- **3 Method**: recap of the WACV formulation in one paragraph (cite it, don't re-derive) → what
  changes for BOP: per-sample K and the `site` depth (derive it the way the paper derives U_z),
  object conditioning, object-balanced sampling, symmetric loss, crops around `bbox_visib` (and the
  BOP +1 px box convention), augmentations (`domain_gap`; why in-plane rotation is invalid when the
  principal point is off-centre).
- **4 Experiments**: 4.1 setup (LM-O, YCB-V PBR → real test; BOP19 metrics: VSD, MSSD, MSPD, AR;
  implementation details; compute budget "4×") → 4.2 comparison (tables in the WACV style, with a
  "Refinement?" and a "Pose from" column instead of "PnP?", plus "Training data" and time) → 4.3
  ablations (the `docs/BOP.md` §2 table split into small tables, final config as one row of each,
  seeds shown) → 4.4 deployment (ONNX/TensorRT, AR fp16 vs PyTorch, latency vs N, detector,
  224 vs 384) → qualitative figure (best / median / worst per dataset, as Figure 3) and failure
  cases (occlusion: eggbox, duck; YCB-V depth bias, as Figure 4).
- **5 Conclusion**: restate, key contribution, limitations (YCB-V real-image depth bias D8,
  occlusion, no Jetson numbers yet), future work (real training images, Jetson, T-LESS).

### B.4 Style checklist (copy the WACV paper's features)
- [ ] "we", active voice; scoped claims ("among PBR-only RGB methods without refinement").
- [ ] Bulleted contributions in the introduction; run-in bold paragraph headings in setup sections.
- [ ] Every symbol defined after its equation; the inverse used at inference written out.
- [ ] Comparison tables with a method-type column, arrows on metrics, "(Ours)", "?" for unknown.
- [ ] The final configuration visible as one row of each ablation table.
- [ ] Hardware and software versions for every latency (A100, CUDA 13, TensorRT 10.14, PyTorch 2.9).
- [ ] A frank paragraph on where we lose (VSD / MSSD gap from the regressed depth, YCB-V without
      real images) turned into a reproducibility argument (released data only, PBR only).
- [ ] Engineering details that show care (box conventions, symmetric clamps, seeds, noise).
- [ ] Figures: teaser, concept figure for the new idea, qualitative best / avg / worst, failure case.
