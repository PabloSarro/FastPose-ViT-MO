"""Debug of the refiner training: input statistics, output scale and gradients over a few steps."""
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.argv = [sys.argv[0], "--dataset", "LMO", "--out_dir", "/tmp/x", "--pixel_aug", "none"]
from src.bop_dataset import *  # noqa
from src.refiner import *  # noqa
from src.refiner import make_input
from src.render import *  # noqa
from src.train_refiner import build_parser, make_datasets, step

args = build_parser().parse_args()
dev = torch.device("cuda")
cfg = SPLITS["lmo"]; ids = list(cfg["obj_ids"]); root = default_bop_root()
assets = load_render_assets(root, "lmo", ids, 65536, default_cache_dir())
centre, half = object_extents(root, "lmo", ids)
renderer = PointRenderer(assets, centre, half, dev)
table, offsets = symmetry_table(root, "lmo", ids)
diam = load_models_info(root, "lmo", ids)["diameter"]
loss_fn = RefineLoss(assets["points"][:, :1024], diam, table, offsets).to(dev)
train, _ = make_datasets(args, assets)
loader = iter(DataLoader(train, batch_size=32, shuffle=True, num_workers=8))
batch = next(loader)
g = lambda k: batch[k].to(dev)
render, _ = renderer.render(g("obj_idx").long(), g("R_init"), g("t_init"), g("K_crop"), 256)
x = make_input(g("image"), render)
print("image mean/std", g("image").mean().item(), g("image").std().item())
print("render mask mean", render[:, 6].mean().item(), "colour mean", render[:, :3].mean().item())
print("channels mean abs:", [round(x[:, c].abs().mean().item(), 3) for c in range(10)])
model = RefinerNet("vit_b_16_256", num_objects=8).to(dev).train()
opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
for it in range(1, 301):
    after, before = step(model, renderer, loss_fn, batch if it == 1 else next(loader), dev, 256)
    opt.zero_grad(); after["total"].backward()
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1e9)
    opt.step()
    if it in (1, 2, 5, 10, 50, 100, 200, 300):
        with torch.no_grad():
            b = next(loader); gg = lambda k: b[k].to(dev)
            r, _ = renderer.render(gg("obj_idx").long(), gg("R_init"), gg("t_init"), gg("K_crop"), 256)
            out = model(make_input(gg("image"), r), gg("obj_idx").long()).float()
        print(f"it {it}: loss {after['total'].item():.4f} grad norm {gn.item():.4f} |out| mean {out.abs().mean().item():.5f} out std over batch {out.std(0).mean().item():.5f}", flush=True)
