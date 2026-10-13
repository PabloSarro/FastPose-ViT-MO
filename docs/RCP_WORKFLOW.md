# Working on FastPose-ViT on the RCP cluster

How I set up the EPFL RCP cluster so that I edit code live from VS Code and run it on a GPU,
with no image build/push after every change. Based on the
[RCP_Quickstart](https://github.com/PierreAncey/RCP_Quickstart) tutorial, adapted to my setup
(Windows laptop + WSL, own Docker image, Pierre's datasets).

Set up and validated on 2026-10-13/14.

---

## 1. The idea

```
Laptop (Windows VS Code) --SSH--> jumphost (no GPU) --runai--> RCP cluster (GPU jobs)
                                       |                              |
                                       +---------- NAS (shared) ------+
```

- Everything lives on the NAS: `/mnt/cvlab/scratch/cvlab/home/sarro` (`$NAS_HOME`).
  The jumphost and every job see the same files, so there is nothing to copy back and forth.
- I **edit** in VS Code connected to the jumphost, and **run** in a GPU job started with `dev`
  (interactive) or `train` (long runs).
- The image only provides dependencies (`cvlab-pablo/fastposevit:1.59`). Code changes need no rebuild.

Old workflow replaced: `runai submit ... --image fastposevit:1.59` from WSL plus `kubectl`.
It still works, but is no longer needed.

> **IMPORTANT: delete jobs when you stop working.** A job holds its GPU until deleted,
> even if idle: `runai delete job fastpose-dev`, then `runai list`.

---

## 2. One-time setup

### 2.1 SSH shortcuts (Windows PowerShell, NOT WSL)

Must be on the EPFL network or VPN. Windows VS Code reads `C:\Users\sarro\.ssh\config`.

```powershell
mkdir -Force "$env:USERPROFILE\.ssh" | Out-Null
ssh-keygen -t ed25519 -f "$env:USERPROFILE\.ssh\id_ed25519_rcp"      # empty passphrase

Add-Content -Encoding ascii "$env:USERPROFILE\.ssh\config" @"

Host rcp
    HostName jumphost.rcp.epfl.ch
    User sarro
    IdentityFile ~/.ssh/id_ed25519_rcp
    RequestTTY yes
    RemoteCommand cd /mnt/cvlab/scratch/cvlab/home/sarro 2>/dev/null; exec bash -l

Host rcp-plain
    HostName jumphost.rcp.epfl.ch
    User sarro
    IdentityFile ~/.ssh/id_ed25519_rcp
"@

# registers the key (asks for the GASPAR password once)
Get-Content "$env:USERPROFILE\.ssh\id_ed25519_rcp.pub" | ssh rcp-plain "mkdir -p ~/.ssh && chmod 700 ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys"

# test: must print "connected to ..." without asking a password
ssh -o BatchMode=yes rcp-plain 'echo connected to $(hostname)'
```

- `rcp`: for a terminal (drops into the NAS folder). `rcp-plain`: for VS Code, `scp`, tunnels.
- The NAS folder already existed, so `mkdir -m 700 -p .../sarro` was a no-op.
- The same entries and key were also copied to the WSL `~/.ssh` (see 7.4), because my VS Code
  read the WSL config path.

### 2.2 Account setup (`ssh rcp`, keep VS Code closed)

```bash
cd /mnt/cvlab/scratch/cvlab/home/sarro
git clone https://github.com/PierreAncey/RCP_Quickstart.git RCP_tutorial
bash RCP_tutorial/setup-jumphost.sh          # [1/5] ... [5/5], "Setup complete"
source "$HOME/.bashrc"                       # adds dev, train, windows, rcp-init
runai login                                  # open the link, paste the code
runai list project
runai config project cvlab-sarro

git config --global user.name "PabloSarro"
git config --global user.email "pablo.sarrosanchez@epfl.ch"
git config --global credential.helper "store --file=$NAS_HOME/.home/.git-credentials"
```

The script moves `~/.vscode-server`, `~/.config`, `~/.cache`, `~/.local`, `~/.bashrc` etc. to
`$NAS_HOME/.home` (home folder inside jobs) and leaves links. See 7.1 if it fails on `~/.local`.

### 2.3 VS Code (Windows, not WSL)

1. Install the **Remote - SSH** extension.
2. In *User Settings (JSON)* make sure:
   ```json
   "remote.SSH.configFile": "C:\\Users\\sarro\\.ssh\\config",
   "remote.SSH.remotePlatform": { "rcp-plain": "linux" }
   ```
   and that there is **no** `"remote.SSH.path": "/usr/bin/ssh"` (see 7.3).
3. `Ctrl+Shift+P` -> *Remote-SSH: Connect to Host...* -> `rcp-plain` (type it if not listed).
4. *File -> Open Folder...* `/mnt/cvlab/scratch/cvlab/home/sarro/FastPose-ViT`.

F1 opens Windows help on this laptop. Use `Ctrl+Shift+P`.

### 2.4 Git over HTTPS

```bash
git remote set-url origin https://github.com/PabloSarro/FastPose-ViT-MO.git
git push           # user: PabloSarro, password: personal access token
```

The token needs the `repo` scope (classic), or Contents: Read and write on this repo
(fine-grained). It is stored once in `$NAS_HOME/.home/.git-credentials`, so it works in jobs too.
SSH remotes fail in jobs (`No user exists for uid 317086`).

---

## 3. Project setup (once per repo)

```bash
cd /mnt/cvlab/scratch/cvlab/home/sarro/FastPose-ViT
rcp-init                       # adds rcp/, AGENTS.md, .claude/settings.json
```

Then edit `rcp/project.env`:

```bash
IMAGE=registry.rcp.epfl.ch/cvlab-pablo/fastposevit:1.59
DATA_DIR=/mnt/cvlab/scratch/cvlab/home/pvancey/datasets     # holds SPEED_FIXED, ...
DEV_JOB=fastpose-dev
GPUS=${GPUS:-1}                # one GPU. Do not put 16 here (see 7.6)
CPUS=${CPUS:-16}
MEMORY=${MEMORY:-32Gi}
NODE_POOL=${NODE_POOL:-default}
```

Results go to `$NAS_HOME/outputs/FastPose-ViT` (`$OUTPUT_DIR`). Then make `outputs` traversable
(needed by `train`, see 7.7):

```bash
chmod 711 /mnt/cvlab/scratch/cvlab/home/sarro/outputs
git add rcp AGENTS.md .gitignore && git commit -m "Add RCP workflow" && git push
```

---

## 4. Daily workflow

1. Open VS Code -> *Open Recent* -> the `[SSH: rcp-plain]` folder (`FastPose-ViT`).
2. Terminal -> New Terminal (this is the jumphost). Start the GPU job:
   ```bash
   dev                         # shell inside job "fastpose-dev" (1 GPU, 16 CPUs, 32 GiB)
   ```
   Wait for `fastpose-dev: Running` and a prompt like `sarro@fastpose-dev:...$`.
3. Edit files in VS Code, run them in the `dev` shell. Changes are visible immediately.
4. When done: `exit`, then on the jumphost:
   ```bash
   runai delete job fastpose-dev
   runai list                  # must be empty
   ```

Other things you can do in `dev`: `windows 4 bash` (grid of 4 terminals), `tmux new -A -s main`
(survives a dropped connection; reconnect with `dev` and `tmux attach`). `dev` is limited to
12 hours and 1 GPU (A100 or V100).

### Environment inside jobs

`$DATA_DIR`, `$OUTPUT_DIR`, `$NAS_HOME`, and for `train` also `$RUN_DIR`. Home is `$NAS_HOME/.home`.
Keep the `PYTHONPATH=.` prefix on every command.

---

## 5. Evaluate the pretrained model (in `dev`)

```bash
DATASET=$DATA_DIR/SPEED_FIXED
RES_DIR=/mnt/cvlab/scratch/cvlab/home/sarro/results/eval-rcp-check
mkdir -p $RES_DIR
PYTHONPATH=. python3 src/evaluate.py \
  --model_weights weights/vit_b_16_384.pth --log_dir $RES_DIR/ \
  --rotation_format matrix --num_hidden_layers 0 --hidden_layer_dim 0 \
  --nb_class_tokens 1 --batch_size 8 --vit_model vit_b_16_384 \
  --dataset SPEED --dataset_root_dir $DATASET --output_json $RES_DIR/results.json \
  --num_workers 8 --merge_outputs --no_mlp
```

Expected (reproduced in the new setup): translation 0.0221, rotation 0.0073, total 0.0294,
**SPEED score 0.0093**. About 15 s. Weights come from Hugging Face
(`PierreAncey/FastPose-ViT`, `vit_b_16_384.pth`) into `weights/` (git-ignored).

---

## 6. Long runs with `train`

`train NAME COMMAND` starts a job that keeps running after logout. Names: lowercase letters,
digits and `-`, never reused. The command runs **without a shell**, so wrap anything with
`$VAR`, `>`, `|`, `&&` or `cd` in `bash -c '...'` (single quotes so variables expand in the job).

Smoke test (2 epochs, about 2 minutes):

```bash
train smoke-02 bash -c 'PYTHONPATH=. python3 src/train.py \
  --model_save_path $RUN_DIR/model.pth --log_dir $RUN_DIR/ \
  --rotation_format matrix --rotation_loss 6d_relative_frobenius \
  --translation_loss relative_translation --max_lr 1e-4 --min_lr 1e-6 --epochs 2 \
  --num_hidden_layers 0 --hidden_layer_dim 0 --nb_class_tokens 1 --batch_size 8 \
  --vit_model vit_b_16_384 --dataset SPEED --dataset_root_dir $DATA_DIR/SPEED_FIXED/ \
  --optimizer muon --scheduler cosineannealingwarmrestarts --num_workers 8 \
  --merge_outputs --no_mlp --no_pixel_augmentation --bbox_crop_percent 10'
```

(Written on several lines here for reading. Paste it as one line.)

Follow it:

```bash
bash rcp/wait.sh smoke-02 done      # waits ~1 min per call; prints the final state
runai list                          # Running -> Succeeded
tail -f $NAS_HOME/outputs/FastPose-ViT/runs/smoke-02/log.txt     # Ctrl+C stops following only
runai delete job smoke-02           # when finished; the run folder is kept
```

Done when the log ends with `Training completed. Best validation loss: ...` and `model.pth`
is in the run folder. The full 310-epoch run is about 9 hours.

Each run lives in `$NAS_HOME/outputs/FastPose-ViT/runs/NAME/`: `log.txt`, `info.txt` (command,
settings, git commit), `code/` (copy of the code that ran), and whatever is saved in `$RUN_DIR`.

Things to remember:

- **Only Git-tracked files are copied** into the run's `code/`. `weights/` and datasets are
  git-ignored, so use absolute paths (`$DATA_DIR/...`) or run with `SNAPSHOT=0` to use the live code.
- Training jobs can be stopped by the cluster and restarted. Save checkpoints in `$RUN_DIR`
  and resume from the last one.
- Resources for one job: `GPUS=2 CPUS=16 MEMORY=64Gi train NAME ...` (2 GPUs only help if the code uses them).
- `bash rcp/stop.sh NAME` or `runai delete job NAME` stops a run.

---

## 7. Troubleshooting (every problem hit during setup)

| # | Symptom | Cause | Fix |
|---|---------|-------|-----|
| 7.1 | `setup-jumphost.sh`: `mv: cannot remove '~/.local/share/claude/versions': Directory not empty` | Claude Code was running on the jumphost from `~/.local`. NFS keeps open files as `.nfs...` | Close Claude on the jumphost. If it still says "Both X and Y exist", `mv ~/.local ~/.local.old` and rerun the script (it links to the NAS copy). Delete `~/.local.old` later. Never run Claude Code on the jumphost, only inside a job. |
| 7.2 | `runai ...: invalid_grant "Offline user session not found"` | Login expired | `runai login` |
| 7.3 | VS Code: `Could not resolve hostname rcp-plain`, `/usr/bin/ssh` not valid | `remote.SSH.configFile` pointed at the WSL path while using Windows `ssh.exe` | Set `configFile` to `C:\\Users\\sarro\\.ssh\\config`, delete `remote.SSH.path`, reload the window |
| 7.4 | `rcp-plain` missing from the host list | VS Code read the WSL config (`/home/pablo/.ssh/config`) | Same fix as 7.3. Typing `rcp-plain` in the picker also works. |
| 7.5 | "Initializing VS Code Server" for 10+ minutes | Half-installed server and stale `agent host` process on the jumphost | `ssh rcp-plain 'kill <PID of code-... agent host>; rm -rf ~/.vscode-server/cli/servers/Stable-<commit> ~/.vscode-server/cli/servers/.locks'` and reconnect |
| 7.6 | `dev`: `blocked ... disallow-more-than-8gpus-pods` | `GPUS=${CPUS:-16}` typo in `rcp/project.env` | `GPUS=${GPUS:-1}`, `CPUS=${CPUS:-16}` |
| 7.7 | `train` -> `StartError`: `mkdir .../outputs/FastPose-ViT: permission denied` | `outputs/` was mode 700. The container runtime checks the run folder as a squashed user | `chmod 711 $NAS_HOME/outputs`, delete the failed job, start with a new name |
| 7.8 | `git push`: `No user exists for uid 317086` | SSH remote inside a job | HTTPS remote + token (2.4) |
| 7.9 | `git push`: `403` | Token without `repo` scope / not allowed on this repo | Recreate the token with the right scope |
| 7.10 | `runai list` shows an old `Deleted` job | Leftover entry | `runai delete job NAME` |

Useful diagnostics: `runai describe job NAME` (Events at the bottom), `runai logs NAME`,
`tail NAME/log.txt` in the run folder. `export SUPPRESS_DEPRECATION_MESSAGE=true` hides the CLI v1 warning.

---

## 8. Command cheat sheet

| Where | Command | What |
|-------|---------|------|
| jumphost | `dev` | shell in GPU job `fastpose-dev` |
| jumphost | `train NAME bash -c '...'` | background training job |
| jumphost | `bash rcp/wait.sh NAME done` | wait for a job to finish |
| jumphost | `runai list` | my jobs |
| jumphost | `runai describe job NAME` | why a job is pending/failed |
| jumphost | `runai delete job NAME` | free the GPU |
| job | `windows 4 bash` | 4 terminals in a grid |
| job | `nvidia-smi` | GPU status |
| any | `rcp-init` | add/update `rcp/` in a repo (keeps `project.env`) |

## 9. Where things are

```
/mnt/cvlab/scratch/cvlab/home/sarro/            ($NAS_HOME)
|- .home/                  settings, tools, logins, shell startup (home inside jobs)
|- RCP_tutorial/           the quickstart scripts (dev, train, ...)
|- FastPose-ViT/           this repo (rcp/, AGENTS.md added)
|- outputs/FastPose-ViT/   runs/NAME/ (log.txt, info.txt, code/, model.pth), jobs.log
\- results/                older manual results
/mnt/cvlab/scratch/cvlab/home/pvancey/datasets/   Pierre's datasets ($DATA_DIR)
```

The NAS is not backed up: keep code in Git and push (`git push`).

## 10. Optional next steps

- Claude Code / Codex inside a job (never on the jumphost):
  `dev`, then `bash "$NAS_HOME/RCP_tutorial/setup-agents.sh"`, then `claude`.
- Use the default lab image `cvlab-pancey/base:0.4` + a `uv` venv instead of my own image.
- More: `RCP_tutorial/guides/` (cluster, agents, jupyter, hybrid laptop/cluster, images).
