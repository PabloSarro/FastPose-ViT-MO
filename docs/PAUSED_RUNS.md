# Paused training runs

Stopped by us to free GPUs for higher-priority runs, to be resumed later. Nothing was deleted: each
checkpoint (`last.pt`, written at the end of every epoch) is in the run folder
`$NAS_HOME/outputs/FastPose-ViT/runs/<stopped job>/<variant>/`.

## Paused 2026-11-25 18:14 UTC (user: low-priority settings and ViT-L 8× seed b)

All are LM-O, 150 epochs. Time left = if it runs without interruption.

| Variant | Change | Seed | Stopped job | Epochs saved | Time left | GPUs | Resumed as |
|---|---|---|---|---|---|---|---|
| `densex5m05` | mask weight 0.5 (`densex5`, 2×) | a | `lmo-m18-densex5m05-001` | 13 | 11.7 h | 1 | |
| `densex5m052` | mask weight 0.5 | b | `lmo-m18-densex5m052-001` | 8 | 11.9 h | 1 | |
| `densex5m20` | mask weight 2 | a | `lmo-m18-densex5m20-001` | 6 | 12.3 h | 1 | |
| `densex5m202` | mask weight 2 | b | `lmo-m18-densex5m202-001` | 2 | 12.6 h | 1 | |
| `densex5m50` | mask weight 5 | a | `lmo-m18-densex5m50-001` | none | 12.8 h | 1 | |
| `densex5m502` | mask weight 5 | b | `lmo-m18-densex5m502-001` | none | 12.8 h | 1 | |
| `densex5r05` | rotation weight 0.5 | a | `lmo-m18-densex5r05-001` | none | 12.8 h | 1 | |
| `densex5r052` | rotation weight 0.5 | b | `lmo-m18-densex5r052-001` | none | 12.8 h | 1 | |
| `densex5r20` | rotation weight 2 | a | `lmo-m18-densex5r20-001` | none | 12.8 h | 1 | |
| `densex5r202` | rotation weight 2 | b | `lmo-m18-densex5r202-001` | none | 12.8 h | 1 | |
| `densex5j15` | box jitter 15 % | a | `lmo-m18-densex5j15-001` | none | 12.8 h | 1 | |
| `densex5j152` | box jitter 15 % | b | `lmo-m18-densex5j152-001` | none | 12.8 h | 1 | |
| `vitl5122` | ViT-L/512 direct regression, 8× | b | `lmo-m15x8-vitl5122-001` | 18 | 112 h | 2 | |

Code at launch: commit `5973ef0` (the 12 `densex5` settings), `f7e33a9` (`vitl5122`).

## How to resume

From the project root. `scripts/resume_run.sh` copies the stopped job's variant folder into the new
run folder (the old one is only read), then runs the stopped job's exact command, which continues
from the copied `last.pt`. Jobs with no saved epoch start from the beginning.

```bash
# 1 GPU (the densex5 settings)
GPUS=1 CPUS=16 MEMORY=32Gi bash rcp/train.sh lmo-m18-densex5m05-002 bash scripts/resume_run.sh lmo-m18-densex5m05-001
# 2 GPUs (ViT-L 8x seed b)
GPUS=2 CPUS=16 MEMORY=96Gi bash rcp/train.sh lmo-m15x8-vitl5122-002 bash scripts/resume_run.sh lmo-m15x8-vitl5122-001
```

Before resuming:
- The new job runs the **current** code. Check `git diff <commit> -- src scripts/m18_dense.sh
  scripts/m10_ablations.sh`: the training must not have changed (up to 2026-11-25 the only changes
  are options that are off by default).
- After the start, its `log.txt` must show `Resumed from ...: N epoch(s) done` with N as above.
- Write the new job name in the "Resumed as" column.
