# Visualisations

Scripts that draw what the pipeline does, so it can be checked by eye. Figures go to
`vis/figures/` (committed, so they are also visible on GitHub). Regenerate all with
`bash vis/run_all.sh`, or run one script from the repo root, e.g. `.venv/bin/python vis/crops.py`.
Every script has `--help`; most take `--seed` (another random sample) and `--out_dir`.

Colours are fixed per object in every figure (ape blue, can orange, cat aqua, driller yellow,
duck pink, eggbox green, glue violet, holepuncher red), and the name is always written too.
Object axes: x red, y green, z blue.

| Figure | Script | What it shows | What to look for |
|---|---|---|---|
| `01_objects.png` | `objects.py` | The 8 LM-O objects: a real LM crop and a synthetic PBR crop each, with BOP id, diameter, symmetry | The synthetic-to-real gap (lighting, texture, blur) |
| `02_scene_train.png`, `02_scene_test.png` | `scene.py` | One full PBR image and one LM-O test image with every instance: `bbox_obj` (dashed), `bbox_visib` (solid), projected model points, axes, % visible | The model points cover each object; `bbox_visib` shrinks when an object is occluded. `--split test --scene 2 --im N` for another image |
| `03_crops_train.png`, `03_crops_test.png` | `crops.py` | Model-input crops exactly as `BOPDataset` returns them (train: jitter + colour augmentation). Dots: GT model points warped into the crop. Axes and cross: the pose **decoded from the labels** with the same inverse functions the predictions go through | Dots on the object, cross on its centre, axes consistent with the dots. Includes eggbox and glue. `--per_object 8` gives the 64-crop grid of the M2 visual check |
| `03_crops_{train,test,val_real}_64.png` | `crops.py --split S --per_object 8 --seed 1` | The M2 visual check: 64 crops per split (8 per object) | Same as above, over many more instances |
| `04_augmentations.png` | `augmentations.py` | The same crops with no augmentation, bbox jitter, `standard` colour (default) and `domain_gap` colour | Why `standard` is the default: `domain_gap` often makes objects unrecognizable (decision D5) |
| `05_reprojection.png` | `reprojection.py` | Top: model and box projected with R vs Rᵀ against `bbox_obj` (the M2 reprojection test). Bottom: pixel-level zoom on the rightmost mask pixels | Top: R matches, Rᵀ doesn't. Bottom: the projected model reaches the far edge of the last mask pixel, so a BOP box `[x, y, w, h]` ends at `x + w + 1` (`bop_bbox_to_xyxy`) |
| `06_index_counts.png` | `stats.py` | Instances per object in each split | Training is balanced (38k–42k per object) |
| `07_index_distributions.png` | `stats.py` | Visibility, box size and distance (train vs test); the translation labels U_x, U_y and U_z in `legacy` vs `site` depth | Test is more occluded and in a narrower distance range than PBR. Legacy U_z is ~100× larger than U_x, U_y; site U_z is around 1 for every object (the M1 design change) |
| `08_speed_golden.png` | `speed.py` | Per-image errors of the published SPEED model (the golden eval, 0.0093) | The reference that every code change must reproduce exactly |
| `09_m1_training_runs.png` | `speed.py` | Short SPEED runs before/after the M1 refactor, and the run killed after epoch 1 and resumed | The refactor and kill/resume runs follow the earlier runs; the learning rate continues exactly |
| `10_lmo_m3_overfit.png` | `training_curves.py RUN_DIR [...]` | Training loss and validation errors per epoch of a run (here the M3 overfit check) | Errors falling towards zero; spikes at the learning-rate warm restarts are expected |
| `11_lmo_m3_ablations.png` | `training_curves.py RUN_DIR ... --labels ...` | Curves of B1, its two repeats and the 3 ablations (legacy depth, decoupled loss, domain_gap aug) | Repeats overlap (noise); legacy converges ~40 epochs late; domain_gap has a higher training loss but better real-image results (table in TASKS.md M3) |
