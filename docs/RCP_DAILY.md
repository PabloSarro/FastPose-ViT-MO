# Daily routine on RCP

Everything is already set up (see [RCP_WORKFLOW.md](RCP_WORKFLOW.md)). This is the short version:
what to do each day, and how to get going again after shutting the laptop down.

---

## 0. What survives a laptop shutdown

Nothing important lives on the laptop. Shutting it down, closing VS Code or losing the connection:

| Thing | Survives? |
|-------|-----------|
| Code, data, results, tools, logins (all on the NAS) | Yes |
| `train` jobs (long runs) | Yes, they keep running without you |
| The `dev` job | The job itself keeps running (and keeps its GPU!), but your terminal in it is gone. Running programs started in a plain shell die with the shell. Use `tmux` (see 3.3) for anything you want to keep. |
| VS Code window, open files, terminals | No. Reopen them (section 1) |
| Run:AI login | For a while. If it expires, run `runai login` again |

So after a restart you do **not** redo the setup. Only reconnect.

A job you forgot keeps its GPU. Before shutting down for the day, always do section 4.

---

## 1. Starting the day (cold start, laptop just booted)

### Step 1. Connect to the EPFL network
- On campus: nothing to do.
- Off campus: start the **EPFL VPN** (e.g. Cisco/OpenConnect client) and wait until it says connected.
  Without it, `rcp-plain` cannot be resolved/reached.

### Step 2. Open VS Code (the Windows one, not from WSL)
Start it from the Start menu (do not run `code .` from the WSL terminal; the window must not
say `WSL: ...` in the bottom-left corner).

### Step 3. Connect to the jumphost
1. `Ctrl+Shift+P` -> **Remote-SSH: Connect to Host...**
2. Pick `rcp-plain`. If it is not in the list, **type `rcp-plain` and press Enter**.
3. A new window opens (first connection of the day can take 10-30 s; a new VS Code version
   reinstalls its server and takes about a minute).
4. **File -> Open Recent** -> `/mnt/cvlab/scratch/cvlab/home/sarro/FastPose-ViT [SSH: rcp-plain]`.
   (Or **File -> Open Folder...** and type the path the first time.)

Shortcut: once connected before, **File -> Open Recent** in a fresh VS Code window lists the
folder directly and connects in one click.

### Step 4. Open a terminal and look around
**Terminal -> New Terminal** (this terminal runs on the jumphost, in the repo folder):

```bash
runai list              # what is running from yesterday? (should be empty or only train jobs)
git status              # clean? on the right branch?
git pull                # only if I pushed from somewhere else
```

If `runai list` complains about `invalid_grant` / "Offline user session not found", log in again:

```bash
runai login             # open the link in the browser, paste the verification code
```

### Step 5. Start a GPU job when I need one
```bash
dev
```
Wait for `fastpose-dev: Running` and the prompt `sarro@fastpose-dev:...$`. If the job already
exists, `dev` just reconnects to it (the first start of a day can take a few minutes while the
image downloads). Typing `dev` in a second terminal gives a second shell in the same job.

That's it: edit in VS Code, run in the `dev` shell.

---

## 2. Checklist version (copy/paste friendly)

```
[ ] On campus or EPFL VPN connected
[ ] Windows VS Code (no "WSL" in the bottom-left corner)
[ ] Ctrl+Shift+P -> Remote-SSH: Connect to Host... -> rcp-plain (type it if missing)
[ ] File -> Open Recent -> FastPose-ViT [SSH: rcp-plain]
[ ] Terminal -> New Terminal; runai list   (runai login if "invalid_grant")
[ ] dev    (when I need the GPU)
```

---

## 3. During the day

### 3.1 Interactive work (`dev`)
```bash
dev                                   # in a VS Code terminal
PYTHONPATH=. python3 src/evaluate.py ...   # in the dev shell; keep the PYTHONPATH=. prefix
```
Edits made in VS Code are visible in the job immediately (same NAS folder). No image rebuild.
Limits: 1 GPU, at most 12 hours. The cluster may remove interactive jobs that stay idle
(idle limit shown by `runai list project` is 2 h), so don't leave `dev` idle.

### 3.2 Long runs (`train`)
From the jumphost terminal in the repo folder (wrap anything with `$VAR`, `>`, `|`, `&&` in `bash -c '...'`):

```bash
train NAME bash -c 'PYTHONPATH=. python3 src/train.py ... --model_save_path $RUN_DIR/model.pth --log_dir $RUN_DIR/ ...'
runai list                                                  # Pending -> Running -> Succeeded
tail -f $NAS_HOME/outputs/FastPose-ViT/runs/NAME/log.txt    # Ctrl+C stops following, not the job
bash rcp/wait.sh NAME done                                  # wait until it finishes
```
- Every run needs a **new** name (lowercase letters, digits, `-`).
- A `train` job keeps running with the laptop off. Next day: `runai list`, `tail` the log.
- Only Git-tracked files are copied into the run; use `$DATA_DIR/...` for data and
  `SNAPSHOT=0 train ...` if the run needs untracked files such as `weights/`.
- When finished or failed: `runai delete job NAME` (the run folder and log are kept).

### 3.3 Keep a program alive if my connection drops
Inside the `dev` shell:
```bash
tmux new -A -s main        # start (or re-attach to) a persistent session
# ... run things ...
# detach: Ctrl+B then D.   Later: dev, then tmux attach -t main
```
Real long runs belong in `train`, not in `dev`.

### 3.4 Save work
```bash
git add -A && git commit -m "..." && git push      # the NAS is NOT backed up
```

---

## 4. Ending the day (do this every time)

```bash
exit                                   # leave the dev shell (if inside it)
runai delete job fastpose-dev          # free the GPU
runai list                             # check: nothing left that I don't need
```
- Delete finished/failed `train` jobs too: `runai delete job NAME`.
- Keep a `train` job only if it is still computing something I want.
- Then close VS Code and shut down the laptop. Nothing else is needed.

---

## 5. If something doesn't work

| Problem | What to do |
|---------|-----------|
| `rcp-plain` not in the host list | Type it manually in the picker. If it fails with "Could not resolve hostname", check VPN and that `remote.SSH.configFile` is `C:\\Users\\sarro\\.ssh\\config` |
| Connection hangs on "Initializing VS Code Server" for more than 3 minutes | Close the window, run in PowerShell: `ssh rcp-plain 'ps -u sarro -o pid,etime,args \| grep "code-" \| grep -v grep'`, kill the old `agent host` PID with `ssh rcp-plain 'kill PID'`, reconnect (details: RCP_WORKFLOW.md 7.5) |
| `ssh rcp-plain` asks for a password | VPN is off, or the key is not loaded. Try `ssh -v rcp-plain` from PowerShell |
| `runai ...: invalid_grant` | `runai login` |
| `dev` stays `Pending` | No free GPU. Wait, or check `runai describe job fastpose-dev` |
| `dev`: policy blocked / "more than 8 GPUs" | Check `GPUS=${GPUS:-1}` in `rcp/project.env` |
| `train` -> StartError, `mkdir ... permission denied` | `chmod 711 $NAS_HOME/outputs`, delete the job, use a new name |
| `git push` rejected / 403 | Token expired or missing `repo` scope: create a new one, then `git push` and enter it as password |
| Anything unclear | `runai describe job NAME` (Events at the bottom) and `runai logs NAME` |

Full troubleshooting table with causes: [RCP_WORKFLOW.md](RCP_WORKFLOW.md), section 7.

---

## 6. Quick reference

| Command | Where | What |
|---------|-------|------|
| `runai list` | jumphost | my jobs |
| `runai login` | jumphost | renew the cluster login |
| `dev` | jumphost | shell in the GPU job `fastpose-dev` |
| `train NAME bash -c '...'` | jumphost | background run |
| `bash rcp/wait.sh NAME done` | jumphost | wait for a run to finish |
| `runai describe job NAME` | jumphost | why a job is pending/failed |
| `runai delete job NAME` | jumphost | free the GPU |
| `windows 4 bash` | in the job | 4 terminals in a grid |
| `tmux new -A -s main` | in the job | session that survives disconnects |
