# Missing pieces before writing the paper

Everything still needed for the follow-up paper on the multi-object BOP work (companion to
`docs/PAPER_FEAT.md`, which describes the paper's structure and style, and `docs/BOP.md`, the draft
of the final tables). Listed 2026-11-08. Tick an item when done and write where its result is
(`TASKS.md` §9 row, run folder, or `docs/BOP.md` section).

Legend: **cost** = what it needs (eval = minutes in `dev`, no training; train = new training
jobs; external = someone outside this project).

**Draft (2026-11-11):** LaTeX draft in `paper/` (Overleaf-ready, see `paper/README.md`); pending results and TODOs are marked in it.

## 1. Results still running
- [x] *Done 2026-11-09: tiny AR 0.645 / 0.650, small 0.654 (default 0.668), `docs/BOP.md`.* **Own multi-class LW-DETR detector**: LM-O AR with our detections vs the default ones (0.668),
      val_pbr mAP, tiny vs small, seed spread. Jobs `lmo-m9-{dettiny,dettiny2,detsmall}-001`.
      Cost: running.
- [x] *Done 2026-11-09, re-measured 2026-11-11: 55 FPS for 8 objects (pipeline 14.1 ms + detector 4.2 ms mean), `docs/BOP.md` §3.* **End-to-end latency with our detector**: TensorRT engine of the detector + pose pipeline,
      per image vs number of objects (the real-time claim of the whole method). Cost: eval, after
      the detector jobs.
- [x] *Done 2026-11-09: 0.643 / 0.643 vs 0.669 / 0.664.* **224 px vs 384 px AR** (D7): LM-O test AR with detections of the 224 px `tok4x` recipe
      (2 seeds) vs `tok4x` / `tok4x2`; latency already measured (pipeline 7.6 vs 14.2 ms at 8
      objects). Jobs `lmo-m10-r224tok4x{,2}-001`. Cost: running, then eval.
- [ ] Optionally, the detector for YCB-V (21 classes), if the LM-O one holds up. Cost: train.

## 2. Evaluation only (no training)
- [x] **Ablations on one metric.** *Done 2026-11-08: `docs/BOP.md` §2, `TASKS.md` §9.* Today they mix driller-only ADD (depth mode, translation loss,
      augmentation, M3), LM real val ADD (conditioning, symmetric loss, compute, M4/M5) and test AR
      (crop source, M7). Re-evaluate the existing `last.pt` of every ablation run with official
      LM-O test AR and the default detections (`scripts/m7_det_eval.sh`), 2 seeds each, so that
      every ablation table uses the same metric. Check first which run folders still have
      `last.pt`. Cost: eval (~3 min per model + official eval).
- [x] **Seed spread of the headline numbers** *Done 2026-11-08: LM-O 0.669 / 0.664 (mean 0.666), YCB-V 0.694 / 0.702.*: LM-O `tok4x2` was never evaluated on test with
      detections; report mean ± spread of the 2 seeds on LM-O (YCB-V has both: 0.694 / 0.702).
      Cost: eval.
- [x] **`model.pth` (best val) vs `last.pt` (final epoch)** *Done 2026-11-08: identical (LM-O 0.6686 vs 0.6685, YCB-V 0.7021 both).*: AR of both for the final models, to
      justify reporting `last.pt` (reasons: one rule for every run, `model.pth` criterion changed
      over time, LM-O validation includes real images, the schedule ends at its LR minimum).
      Cost: eval.
- [x] **Per-object official AR** *Done 2026-11-08: `docs/BOP.md` §2.1, `scripts/bop_per_object_ar.py`.* on LM-O and YCB-V (today only own per-object recalls exist in
      `table_test.md`). Cost: eval.
- [x] **Model size and compute table** *Done 2026-11-08: `docs/BOP.md` §2.2 (86.1 M, 55.5 GMACs at 384 px; matches WACV Table 1).* (parameters, GFLOPs at 224 and 384 px, engine size), as
      Table 1 of the WACV paper. Cost: eval.
- [x] **Detection quality** *Default detections done 2026-11-08 (`docs/BOP.md` §2.3, `scripts/detection_map.py`); ours when the detector jobs finish.*: default detector (GDRNPPDet) vs ours, mAP / recall on the BOP test
      targets, to discuss what the detector costs in AR. Cost: eval.
- [x] *Done 2026-11-08: re-timed, LM-O 134 → 98.5 ms, YCB-V 118 → 93.6 ms incl. detector, AR unchanged; `docs/BOP.md` §1 updated.* Check that the M6–M8 BOP `time` columns are explained (old pipeline path, slower than the
      M9 path) or recomputed with the current `estimate_image`. Cost: eval.

## 3. Studies not done on BOP (need training)
- [~] *Added 2026-11-08 (user + supervisor):* **augmentation study**: no pixel aug / color / blur / noise / color + blur + noise / domain_gap at p = 0.8, 2 seeds each (`lmo-m10-aug*-001`), plus the bbox jitter values below. **On hold since 2026-11-08 (cluster queue, user priority 4 of 4): relaunch command in `TASKS.md` M10.**
- [x] *Done 2026-11-09: absolute 0.546 / 0.548 vs apparent 0.655 / 0.655 (`docs/BOP.md` §2).* **Apparent (allocentric) rotation vs absolute rotation** on BOP: the WACV paper's core
      contribution; reviewers will ask whether it still helps here. Cost: train (2 seeds).
- [x] *Done 2026-11-11: scratch 0.516 / 0.517 vs SWAG 0.655 (`docs/BOP.md` §2); plain ImageNet weights not run (no 384 px ImageNet-1K ViT-B in torchvision).* **Pre-training**: SWAG vs plain ImageNet weights vs from scratch. Cost: train.
- [~] *0 % 0.625, 20 % 0.650 vs 10 % 0.655 (done 2026-11-11); 5 / 15 / 30 % queued (`lmo-m10-bbox{5,15,30}{,2}-002`).* **Bounding-box perturbation**: 0 / 10 / 20 % (the WACV paper's Table 8, on BOP). Cost: train.
- [x] *Done 2026-11-11 on LM-O, 8 objects: `legacy` 0.610 / 0.625 vs `site` 0.655; YCB-V not run.* **`site` vs `legacy` depth on several objects / cameras** (so far: driller only, one
      camera): multi-object LM-O and/or YCB-V (two cameras), to support "camera- and
      object-size-independent". Cost: train.
- [ ] Optionally, **object conditioning on YCB-V** (heads vs class token with 21 objects).
      Cost: train.

## 4. Needs outside input or resources
- [ ] **Jetson Orin Nano latency** for BOP (the WACV paper's central claim was on-device; we only
      have A100). Needs device access (supervisor question in `TASKS.md` §8). Cost: external.
- [x] *Done 2026-11-11: every LM-O / YCB-V entry checked on its BOP submission page (training data,
      number of models, refinement); tables rebuilt in `paper/sections/experiments.tex`. Finding:
      PViT-6D (0.740 LM-O, PBR only) and PoseDETR (0.695, 12 ms) are more accurate on LM-O; PViT-6D
      already published our `site` depth (SIZP) and a shared class-conditioned model, so the
      contributions were reframed (see the draft's TODOs; positioning to agree with the
      supervisor).* **Verify the leaderboard entries by hand** on the BOP website (the numbers were read with an
      automatic extractor): AR values, training data, and refinement for MRC-Net, PoseDETR and SC6D
      (marked "?"). Note that competitors' times come from different hardware: say so or re-time
      some of them. Cost: external / manual.
- [x] *Done 2026-11-11 for the recent entries (FastPose-ViT, LW-DETR, GDRNPP, PViT-6D, MRC-Net,
      SC6D, RADet, PFA, Muon); classic entries spot-checked 2026-11-12 (BOP 2018 / 2020,
      BlenderProc, SWAG, ZebraPose, MegaPose, T-LESS, LM: all matched; pages added). MRPE, PoseDETR, RACE-6D (CVPR 2026
      Findings) have no citable paper in the bib yet.* **Bibliography**: exact references for every compared and related method (ZebraPose,
      GDRNPP, SurfEmb, CosyPose, CDPN, EPOS, MRC-Net, PoseDETR, SC6D, PViT-6D, MegaPose, LW-DETR,
      BOP challenge papers, …). Cost: manual.
- [ ] **Venue, page limit, authorship** (supervisor question "expected deliverable"). Cost: external.
- [ ] **Release plan**: BOP weights (e.g. Hugging Face, as for the WACV weights) and code; the
      paper argues reproducibility. Cost: external decision, then upload.

## 5. Scope decisions (user)
- [x] *Decided 2026-11-08: in. Data, tree, index (+ multi-instance fix), overfit check (all 30 objects 1.000) done; jobs `tless-m10-tok4x{,2}-002` queued 2026-11-08 ~23:20 (priority 3), ~83 h each.* **T-LESS**: in or out? Most BOP papers report LM-O, T-LESS and YCB-V; T-LESS also brings
      several instances of the same object (D6). Cost: train if in.
- [~] *2026-11-08: the user asks the supervisor; meanwhile the missing `.z01` was downloaded and the data extracted to `$NAS_HOME/datasets/ycbv_train_real/`. Training with it not decided.* **YCB-V `train_real` (D8)**: fix the real-image depth bias (+2.4 % median) by training on
      PBR + real, or keep the PBR-only result as a stated limitation. Cost: download 48 GB + train
      if in.

## 6. Figures (none made yet)
- [x] *Done 2026-11-11 (`paper/figures/`, scripts `vis/paper_*.py`).* **Teaser**: one image → detector → N crops batched through one shared ViT with per-object
      class tokens → N poses.
- [ ] **Concept figure** for the main new idea (class-token conditioning and/or the `site` depth).
- [x] *Done 2026-11-11 (`paper/figures/`, scripts `vis/paper_*.py`).* **Qualitative results**: best / median / worst predictions per dataset (GT axes solid,
      predictions dashed, errors under each image), as Figure 3 of the WACV paper.
- [x] *Done 2026-11-11 (`paper/figures/`, scripts `vis/paper_*.py`).* **Failure cases**: occlusion (LM-O eggbox, duck) and the YCB-V depth bias, as Figure 4.
- [x] *Done 2026-11-11 (`paper/figures/`, scripts `vis/paper_*.py`).* **Latency vs number of objects** plot (PyTorch vs TensorRT, 224 vs 384 px, with and without
      the detector).
- Reuse `vis/` (`bash vis/run_all.sh`, `vis/training_curves.py`) where possible.

## Suggested order
1. Section 2 (eval only, ~1 h of `dev`, while the section 1 jobs run).
2. Section 1 as the jobs finish; update `docs/BOP.md`.
3. Section 5 decisions, then section 3 jobs in parallel (one per config, 2 seeds, as usual).
4. Section 6 figures and section 4 items alongside.
