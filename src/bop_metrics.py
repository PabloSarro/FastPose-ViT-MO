# This file computes BOP pose errors on the GPU during validation (M5): ADD(-S), MSSD and MSPD per
# instance, with each object's symmetries, and the recalls used for model selection.
#
# Same definitions as bop_toolkit_lib.pose_error (all vertices of models_eval, the BOP19 symmetry
# set with its translation offsets), so the values match src/bop_evaluate.py; the BOP19 AR itself
# (with VSD) comes from scripts/bop_eval.sh.

import os

import numpy as np
import torch
from bop_toolkit_lib import inout, misc

# BOP19 error thresholds: MSSD in fractions of the diameter, MSPD in pixels at 640 px width
MSSD_THRESHOLDS = torch.arange(1, 11) * 0.05
MSPD_THRESHOLDS = torch.arange(1, 11) * 5.0


class BOPPoseMetrics:
    """Per-instance ADD(-S), MSSD and MSPD of a batch of absolute poses, accumulated over an epoch.

    Usage: reset(), update(...) for every batch, compute() at the end.
    """

    def __init__(self, bop_root: str, dataset: str, obj_ids, device: torch.device,
                 max_sym_disc_step: float = 0.01):
        models_dir = os.path.join(bop_root, dataset, "models_eval")
        info = inout.load_json(os.path.join(models_dir, "models_info.json"), keys_to_int=True)
        pts = [
            inout.load_ply(os.path.join(models_dir, f"obj_{o:06d}.ply"))["pts"] for o in obj_ids
        ]
        syms = [misc.get_symmetry_transformations(info[o], max_sym_disc_step) for o in obj_ids]
        n_pts = max(len(p) for p in pts)
        n_sym = max(len(s) for s in syms)
        # Padding repeats the first vertex / the identity: no effect on a max or min; the ADD mean
        # uses the mask
        pts_table = np.stack([np.concatenate([p, np.repeat(p[:1], n_pts - len(p), 0)]) for p in pts])
        mask = np.stack([np.arange(n_pts) < len(p) for p in pts])
        sym_R = np.tile(np.eye(3), (len(obj_ids), n_sym, 1, 1))
        sym_t = np.zeros((len(obj_ids), n_sym, 3))
        for i, s in enumerate(syms):
            sym_R[i, : len(s)] = [x["R"] for x in s]
            sym_t[i, : len(s)] = [x["t"][:, 0] for x in s]

        def t(x, dtype=torch.float32):
            return torch.as_tensor(np.asarray(x), dtype=dtype, device=device)

        self.obj_ids = tuple(int(o) for o in obj_ids)
        self.pts = t(pts_table)  # (n_obj, P, 3) mm
        self.pts_list = [t(p) for p in pts]  # unpadded, for ADD-S
        self.mask = t(mask)  # (n_obj, P)
        self.sym_R, self.sym_t = t(sym_R), t(sym_t)  # (n_obj, K, 3, 3), (n_obj, K, 3)
        self.diameter = t([info[o]["diameter"] for o in obj_ids])
        self.symmetric = torch.as_tensor([len(s) > 1 for s in syms], device=device)
        self.reset()

    def reset(self) -> None:
        self._errs = {k: [] for k in ("obj_idx", "add", "mssd", "mspd", "img_w")}

    @torch.no_grad()
    def errors(self, R_est, t_est, R_gt, t_gt, K, obj_idx) -> dict:
        """Per-instance errors (mm, mm, px) of absolute poses x_c = R·x_m + t (t in mm).

        Args:
            R_est, R_gt: (B, 3, 3); t_est, t_gt: (B, 3) mm; K: (B, 3, 3); obj_idx: (B,)
        """
        R_est, t_est, R_gt, t_gt, K = (x.float() for x in (R_est, t_est, R_gt, t_gt, K))
        pts = self.pts[obj_idx]  # (B, P, 3)
        est = pts @ R_est.transpose(1, 2) + t_est[:, None]  # (B, P, 3)
        # Symmetric GT poses: R_gt·S_k, R_gt·t_k + t_gt
        R_sym = R_gt[:, None] @ self.sym_R[obj_idx]  # (B, K, 3, 3)
        t_sym = (R_gt[:, None] @ self.sym_t[obj_idx][..., None])[..., 0] + t_gt[:, None]
        gt = pts[:, None] @ R_sym.transpose(2, 3) + t_sym[:, :, None]  # (B, K, P, 3)

        mssd = (est[:, None] - gt).norm(dim=-1).amax(dim=-1).amin(dim=-1)

        def project(x):
            x = x @ K.transpose(1, 2).reshape(K.shape[0], *([1] * (x.dim() - 3)), 3, 3)
            return x[..., :2] / x[..., 2:3]

        mspd = (project(est)[:, None] - project(gt)).norm(dim=-1).amax(dim=-1).amin(dim=-1)

        # ADD for asymmetric objects (identity GT), ADD-S for symmetric ones
        mask = self.mask[obj_idx]
        add = ((est - gt[:, 0]).norm(dim=-1) * mask).sum(-1) / mask.sum(-1)
        for i in torch.nonzero(self.symmetric[obj_idx]).flatten().tolist():
            o = int(obj_idx[i])
            p = self.pts_list[o]
            e = p @ R_est[i].T + t_est[i]
            g = p @ R_gt[i].T + t_gt[i]
            add[i] = torch.cdist(e, g).amin(dim=0).mean()  # GT point -> nearest estimated, as adi
        return {"add": add, "mssd": mssd, "mspd": mspd}

    @torch.no_grad()
    def update(self, R_est, t_est, R_gt, t_gt, K, obj_idx, img_w) -> None:
        obj_idx = obj_idx.to(self.pts.device).long()
        errs = self.errors(R_est, t_est, R_gt, t_gt, K, obj_idx)
        for k, v in errs.items():
            self._errs[k].append(v)
        self._errs["obj_idx"].append(obj_idx)
        self._errs["img_w"].append(img_w.to(self.pts.device).float())

    def compute(self) -> dict:
        """Recalls over all accumulated instances, as src/bop_evaluate.py and BOP19.

        Returns:
            AR_MSSD, AR_MSPD, score = (AR_MSSD + AR_MSPD) / 2 (model selection), ADD(-S)@0.1d,
            n, and per-object AR_MSSD/AR_MSPD (key "per_object": {obj_id: (mssd, mspd)}).
        """
        e = {k: torch.cat(v) for k, v in self._errs.items()}
        d = self.diameter[e["obj_idx"]]
        th = MSSD_THRESHOLDS.to(d.device)
        mssd_hit = (e["mssd"][:, None] < th * d[:, None]).float().mean(1)
        mspd_scaled = e["mspd"] * 640.0 / e["img_w"]
        mspd_hit = (mspd_scaled[:, None] < MSPD_THRESHOLDS.to(d.device)).float().mean(1)
        out = {
            "n": int(len(d)),
            "AR_MSSD": float(mssd_hit.mean()),
            "AR_MSPD": float(mspd_hit.mean()),
            "ADD@0.1d": float((e["add"] < 0.1 * d).float().mean()),
            "per_object": {},
        }
        out["score"] = (out["AR_MSSD"] + out["AR_MSPD"]) / 2
        for i, o in enumerate(self.obj_ids):
            m = e["obj_idx"] == i
            if m.any():
                out["per_object"][o] = (float(mssd_hit[m].mean()), float(mspd_hit[m].mean()))
        return out
