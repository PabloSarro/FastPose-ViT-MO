# rcp/

These scripts run this project on the EPFL RCP cluster. Once the RCP tutorial is set up, the `dev` and `train` commands call them for you from any folder in the project. To add this folder to another repository, or to update it, run `rcp-init` from that repository.

| File | What it is for |
| --- | --- |
| `project.env` | Image, folders and job size. This is the only file you edit. |
| `interactive.sh` | Starts your interactive job (`dev`) and opens a shell in it. Run it on the jumphost. |
| `train.sh` | Starts a training job that keeps running after you log out. |
| `wait.sh` | Waits until a job is running or finished. |
| `stop.sh` | Deletes one of this project's training jobs. Its output is kept. |
| `logged.sh`, `common.sh` | Helpers for the other scripts. |
| `agent-instructions.md` | Notes for coding assistants. Add them to `AGENTS.md`, which Codex and Claude Code both read. |
| `claude-settings.json` | Claude Code permissions. Lets the assistant run job commands without asking, and blocks the usual ways of deleting jobs or data, plus Git commands that discard work. |

---

## Folders

By default, a project uses these folders:

```
$NAS_HOME/datasets/                 DATA_DIR    input data, shared by your projects
$NAS_HOME/outputs/PROJECT/          OUTPUT_DIR  results of this project
├── jobs.log                        every training job started or stopped
└── runs/NAME/                      one folder per training job (RUN_DIR inside the job)
    ├── log.txt                     everything the job printed
    ├── info.txt                    command, settings, Git commit
    └── code/                       copy of the code the job runs
```

You can change them in `project.env`. Jobs get `DATA_DIR`, `OUTPUT_DIR` and, for training jobs, `RUN_DIR` as environment variables. Read your data from `$DATA_DIR` and save results in `$RUN_DIR`. In a Git worktree, `PROJECT` is still the main repository's name, so all worktrees share one output folder.

Inside jobs, your home folder is `$NAS_HOME/.home`, so your tools (uv, Claude Code, Codex) and their logins carry over from one job to the next. Every time you start a job from the jumphost, the scripts also copy the jumphost's `runai` there and pass on your Run:AI login. That way `train.sh`, `wait.sh` and `stop.sh` work inside jobs too. If `runai` can't reach the cluster from a job, they print the command to run on the jumphost instead.

---

## Use

```bash
bash rcp/interactive.sh                                   # shell in the job "dev" (on the jumphost)
bash rcp/train.sh lr-001 .venv/bin/python train.py        # training job "lr-001"
bash rcp/wait.sh lr-001 done                              # wait for it (1 minute per call)
tail -n 50 RUN_FOLDER/log.txt                             # its output (train.sh prints the run folder)
bash rcp/stop.sh lr-001                                   # delete it; the run folder is kept
```

A training job runs a copy of the code, so editing files afterwards doesn't affect it. Only files Git would commit get copied, and `.venv` is linked. Set `SNAPSHOT=0` to run the live code instead.

Job names can only contain lowercase letters, digits and `-`, and every run needs a new one. To change the size for a single job, put it in front of the command: `GPUS=2 CPUS=16 MEMORY=64Gi bash rcp/train.sh ...`.

Delete jobs as soon as you're done with them, because an idle job still holds its GPU. To delete `dev`, run `runai delete job dev` on the jumphost.

When things go as expected, the scripts use these exit codes: 0 ok, 1 job failed, 2 no such job, 3 still waiting, 4 runai not usable or the job could not be started, 5 invalid request.
