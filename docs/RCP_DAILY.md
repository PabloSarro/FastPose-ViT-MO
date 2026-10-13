# Daily routine on RCP

Setup is done (see [RCP_WORKFLOW.md](RCP_WORKFLOW.md)). After a shutdown, only reconnect.

## Connect VS Code

1. EPFL VPN on (unless on campus).
2. Open the Windows VS Code (no `WSL` in the bottom-left corner).
3. `Ctrl+Shift+P` -> **Remote-SSH: Connect to Host...** -> `rcp-plain` (type it if not listed).
4. **File -> Open Recent** -> `FastPose-ViT [SSH: rcp-plain]`.
5. **Terminal -> New Terminal**.

No commands are needed to start working: the code is already on the NAS. If a command says
`invalid_grant`, run `runai login`.

## Launch a job

```bash
dev                                   # interactive GPU shell (job fastpose-dev)
train NAME bash -c '<command>'        # long run in the background (new NAME each time)
```

## When done

```bash
exit
runai delete job fastpose-dev
runai list                            # must be empty
```
