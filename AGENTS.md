## Running on the RCP cluster

You run inside an RCP job named `$RCP_JOB` (usually `dev`): 1 GPU, 12 hours at most. Run tests and short experiments here directly. Anything longer goes to a training job. Run the scripts as `bash rcp/...` from the project root (the folder that contains `rcp/`), because the permission rules expect exactly that.

**Folders:**
- `$DATA_DIR`: input data. Read only: never modify or delete anything there.
- Run folders: each training job gets one, `.../outputs/PROJECT/runs/NAME/`. `train.sh` prints its exact path ("Run folder:"). Use that path. Inside the job it is `$RUN_DIR`: save results and checkpoints there. It holds `log.txt` (everything the command printed, kept after the job is deleted), `info.txt` (what was run) and `code/` (the copy of the code the job runs). Never delete anything in it. `jobs.log`, one level up, lists every job started or stopped.

**Commands:**

| Command | What it does |
| --- | --- |
| `bash rcp/train.sh NAME COMMAND` | Start a training job on its own GPU. Prints its run folder. |
| `bash rcp/wait.sh NAME done` | Wait up to 1 minute for it to finish (returns 3 if still running). |
| `tail -n 50 RUN_FOLDER/log.txt` | Read its output, also after it ended. |
| `runai logs NAME` | Startup errors that happen before your command runs (empty `log.txt`). |
| `runai list` · `runai describe job NAME` | Status of all jobs · why one is pending or failed. |
| `bash rcp/stop.sh NAME` | Delete a training job. Its run folder is kept. |

Exit codes: 0 ok · 1 job failed · 2 no such job · 3 still waiting · 4 runai not usable, or the job could not be started (the message says why and what to ask the user) · 5 invalid request, e.g. a bad or already used name.

**Rules:**
- GPUs are limited and shared by the lab. Stop each training job as soon as it has finished, failed or is no longer needed. Run one at a time unless asked. Use the smallest size that works. The default is 1 GPU, 8 CPUs and 32 GiB. Change it with `GPUS=2 CPUS=16 MEMORY=64Gi bash rcp/train.sh ...` (`GPUS=0` for CPU only).
- Waiting: for long runs, check rarely (`WAIT_MINUTES=9 bash rcp/wait.sh NAME done`, with a 10-minute command timeout). If a job is still Pending after about 15 minutes, tell the user and suggest a smaller size.
- Never stop `$RCP_JOB`, the job you run in, and never stop jobs you did not start.
- Never run Git commands that discard work: `git reset --hard`, `git clean`, `git push --force`, `git branch -D`, `git checkout -- …`, `git restore`, `git stash drop`. Commit often on your own branch instead, and ask the user before anything destructive.
- Job names are new each time: lowercase letters, digits and `-`, at most 40 characters. In a Git worktree, start them with the worktree name (e.g. `renderer-lr-001`).
- In a new Git worktree, first create its Python environment: `uv venv --system-site-packages && uv pip install -r requirements.txt`.
- COMMAND runs from the root of the code copy, without a shell. For `>`, `|`, `&&`, `cd` or `VAR=...`, wrap it: `bash rcp/train.sh NAME bash -c "cd sub && python run.py > out.txt"`.
- Files ignored by Git are not in the code copy: read data from `$DATA_DIR`.
- The cluster can stop and restart training jobs: save checkpoints in `$RUN_DIR` and resume from the latest one.
- Python: use `.venv/bin/python` (no activation needed). Install packages with `uv pip install PACKAGE`, from this job, and not while a training job uses the environment.
- Keep your own notes (what you changed, jobs running, where results are) in a separate `## Project notes` section of this file, not in this one: this section is replaced when the RCP scripts are updated. The session ends after 12 hours and the next one starts from your notes.

## Project notes

**What this is:** FastPose-ViT is a ViT that directly regresses the 6D pose of one spacecraft from a bbox crop of one RGB image (SPEED/SPEED+). The current project extends it to **multi-object 6D pose estimation on BOP** (LM-O first, then YCB-V), measured with bop_toolkit BOP19 AR.
- Constraints: RGB only, direct regression (no PnP or refinement), Jetson real-time, and SPEED must keep working.
- **Read `TASKS.md` §1 (status) and §10 (next steps) at the start of every session, and update §1, §9 (results) and §10 before ending.**

**Data** (read only):
- `$DATA_DIR/SPEED_FIXED` etc.
- BOP datasets in `$DATA_DIR/BOP_Datasets/<ds>/<ds>_<split>/<split>/<scene>` (nested layout; build symlinked standard trees under `$NAS_HOME/datasets/bop/`, never inside `$DATA_DIR`).
- Pretrained weights: `weights/vit_b_16_384.pth` (git-ignored, so not in `train` snapshots: pass absolute paths).

**Key commands** (always prefix `PYTHONPATH=.`):
- SPEED golden eval (must print SPEED score 0.0093; reference `$NAS_HOME/results/eval-rcp-check/results.json`): see `docs/RCP_WORKFLOW.md` §5.
- Training: `bash rcp/train.sh NAME bash -c 'PYTHONPATH=. .venv/bin/python src/train.py ... --log_dir $RUN_DIR/ --model_save_path $RUN_DIR/model.pth --resume auto'`, with the best SPEED flags in `README.md`. With `--resume auto`, a preempted/restarted job continues from `$RUN_DIR/last.pt` (~690 MB, saved every epoch).
- BOP training: same command with `--dataset LMO` (data from the cached index; `--bop_obj_ids 8` for one object, `--bop_overfit 256` for the sanity check, batch 32). Plot curves: `.venv/bin/python vis/training_curves.py RUN_DIR`.
- Tests: `.venv/bin/python -m pytest tests/`.
- Figures: `bash vis/run_all.sh` → `vis/figures/` (see `vis/README.md`). Regenerate after changing data or label code.
- BOP tree: `bash scripts/make_bop_tree.sh` (→ `$NAS_HOME/datasets/bop`). Official BOP eval: `bash scripts/bop_eval.sh RESULTS_DIR METHOD_DS-SPLIT.csv` (handles the missing-libEGL workaround, venv PATH and `BOP_PATH`).

**Conventions:**
- Code work on branch `multi-object`. After each §7 bullet: commit with a 2–4 word Title Case message (e.g. `BOP Access Verified`), no `Co-Authored-By: Claude` trailer (author is PabloSarro only), and `git push origin multi-object` (fork `PabloSarro/FastPose-ViT-MO`; never push to `upstream`).
- Job names prefixed by dataset/milestone, e.g. `lmo-m3-b1-001`.
- Experiments (user, 2026-10-22): treat training GPUs as effectively unlimited. Don't pick one config by assumption: when a choice is open, launch one train job per candidate config in parallel (same budget as the baseline, e.g. 384 px / 150 epochs), include repeats of the baseline to measure run-to-run noise, and decide from the results. Precedent: M3 ablations (`scripts/m3_ablations.sh`, one variant per job).
- New work found along the way (a prerequisite, a fix, an extra check): add it as a new bullet in `TASKS.md` §7 under the right milestone (with a date) instead of doing it silently or skipping it.
- RCP quirks (2026-10-20):
  - A new job normally appears in `runai list` within ~10 s. Once the cluster delayed new jobs by ~9 min, and `wait.sh` returned 2 in the meantime, so re-check for a few minutes before concluding a job failed. `$RUN_DIR/log.txt` ("exit code" line) is the reliable signal.
  - Put job logic in script files, not inline multi-line `python -c`, because runai strips the inner quotes.
  - Install packages with `.venv/bin/pip`: uv ignores the system site-packages.
- Results of BOP runs go in `$RUN_DIR`, and a summary row goes in `TASKS.md` §9.
- Geometry conventions:
  - BOP `cam_R_m2c` is used as is: it is the same R model→camera as SPEED's `quaternion_to_rotation_matrix(q_vbs2tango)`.
  - BOP translations are in mm.
  - The image-rotation augmentation is invalid for BOP, because the principal point is not the image centre.
- Keep this file as the only agent-instructions file. Don't add `CLAUDE.md`/`CLAUDE.local.md` unless its first line is `@AGENTS.md`.
