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
