"""Save a montage of refiner samples: image crop | render at the starting pose | render at the GT
pose | image with the GT render's mask outline in green and the starting one in red."""
import sys

import cv2
import numpy as np
import torch

sys.argv = [sys.argv[0], "--dataset", "LMO", "--out_dir", "/tmp/x", "--pixel_aug", "none", "--noise", sys.argv[1] if len(sys.argv) > 1 else "mix"]
from src.bop_dataset import *  # noqa
from src.refiner import *  # noqa
from src.render import *  # noqa
from src.train_refiner import build_parser, make_datasets

args = build_parser().parse_args()
dev = torch.device("cuda")
ids = list(SPLITS["lmo"]["obj_ids"]); root = default_bop_root()
assets = load_render_assets(root, "lmo", ids, 65536, default_cache_dir())
centre, half = object_extents(root, "lmo", ids)
renderer = PointRenderer(assets, centre, half, dev)
train, _ = make_datasets(args, assets)
mean = np.array(IMAGENET_MEAN); std = np.array(IMAGENET_STD)
rows = []
for i in np.random.default_rng(1).choice(len(train), 6, replace=False):
    s = train[int(i)]
    img = (s["image"].permute(1, 2, 0).numpy() * std + mean).clip(0, 1)
    obj = torch.tensor([s["obj_idx"]]).to(dev)
    K = s["K_crop"][None].to(dev)
    r0, _ = renderer.render(obj, s["R_init"][None].to(dev), s["t_init"][None].to(dev), K, 256)
    r1, _ = renderer.render(obj, s["R_gt"][None].to(dev), s["t_gt"][None].to(dev), K, 256)
    c0 = r0[0, :3].permute(1, 2, 0).cpu().numpy(); c1 = r1[0, :3].permute(1, 2, 0).cpu().numpy()
    over = (img * 255).astype(np.uint8).copy()
    for r, colour in ((r1, (0, 255, 0)), (r0, (255, 0, 0))):
        m = (r[0, 6].cpu().numpy() > 0.5).astype(np.uint8)
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(over, cs, -1, colour, 1)
    rows.append(np.concatenate([(img * 255).astype(np.uint8), (c0 * 255).clip(0, 255).astype(np.uint8), (c1 * 255).clip(0, 255).astype(np.uint8), over], axis=1))
cv2.imwrite(sys.argv[-1] if False else "/mnt/cvlab/scratch/cvlab/home/sarro/results/m19_smoke/samples.png", cv2.cvtColor(np.concatenate(rows, axis=0), cv2.COLOR_RGB2BGR))
print("saved")
