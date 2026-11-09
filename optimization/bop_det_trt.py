# M9 stretch: TensorRT engine of the multi-class BOP LW-DETR (object_detector/bop_detector.py),
# for the end-to-end latency (detector + pose) and the AR with detections from the engine.
#
# Steps (one command, from the repo root):
#   1. ONNX export of the trained detector (best.pth) with the LW-DETR export path: static
#      spatial shapes, last decoder layer only, one query group, deformable-attention sampling
#      with gathers instead of GridSample (TensorRT 10.14's GridSample is wrong for it); input images (1, 3, S, S)
#      float32 (ImageNet-normalised), outputs pred_boxes (1, Q, 4) cxcywh in [0, 1] and
#      pred_logits (1, Q, C).
#   2. TensorRT engine (fp16 by default), batch 1.
#   3. Engine vs PyTorch fp32 on real test images (max |diff| of boxes and sigmoid scores).
#   4. Detections of the whole split with the engine (same pre- and post-processing as
#      bop_detector.Detector, timed per image incl. resize, normalisation and top-k), written in
#      the BOP default-detections format: OUT/lwdetr_trt_<dataset>-<split>.json (+ _summary.json).
#
#   PYTHONPATH=. .venv/bin/python optimization/bop_det_trt.py --variant small \
#       --weights RUN/small/best.pth --out_dir OUT

import argparse
import json
import logging
import os

import cv2
import numpy as np
import tensorrt as trt
import torch

from object_detector.LWDETR.lwdetr_ops.functions import ms_deform_attn_func
from object_detector.bop_detector import BOPDetDataset, BOPLWDETR, Detector, detect_split
from optimization.bop_trt import TRT_LOGGER, build_engine
from src.bop_dataset import SPLITS, load_index
from src.bop_evaluate import normalize_crops

os.environ.setdefault("USER", str(os.getuid()))  # torch.export needs getpass.getuser()


def export_model(model: BOPLWDETR, img_size: int) -> torch.nn.Module:
    """The LW-DETR module in its export mode, with static spatial shapes for `img_size`."""
    net = model.lwdetr_model.eval()
    base = img_size // 16  # ViT patch size
    scale = {"P3": 2.0, "P4": 1.0, "P5": 0.5, "P6": 0.25}
    levels = model._create_single_object_config(model.variant).projector_scale
    hw = [(round(base * scale[lvl]),) * 2 for lvl in levels]
    net.transformer.static_spatial_shapes = torch.tensor(hw, dtype=torch.long)
    net.transformer.static_spatial_shapes_list = hw
    net.export()  # also switches every sub-module with an export() method
    ms_deform_attn_func.GATHER_SAMPLER = True  # TensorRT's GridSample is wrong here (see there)
    return net


class TRTDetector(Detector):
    """bop_detector.Detector with the network replaced by a TensorRT engine (batch 1)."""

    def __init__(self, engine_path: str, post, img_size: int, obj_ids, top_k: int):
        self.img_size, self.obj_ids, self.post = img_size, obj_ids, post
        post.num_select = top_k
        self.runtime = trt.Runtime(TRT_LOGGER)
        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.shapes = {n: tuple(self.engine.get_tensor_shape(n)) for n in names}
        self.stream = torch.cuda.Stream()

    def raw(self, x: torch.Tensor) -> dict:
        x = x.float().contiguous()
        boxes = torch.empty(self.shapes["pred_boxes"], device="cuda")
        logits = torch.empty(self.shapes["pred_logits"], device="cuda")
        for name, t in (("images", x), ("pred_boxes", boxes), ("pred_logits", logits)):
            self.context.set_tensor_address(name, t.data_ptr())
        caller = torch.cuda.current_stream()
        self.stream.wait_stream(caller)
        if not self.context.execute_async_v3(self.stream.cuda_stream):
            raise RuntimeError("TensorRT execution failed")
        caller.wait_stream(self.stream)
        for t in (x, boxes, logits):
            t.record_stream(self.stream)
        return {"pred_boxes": boxes, "pred_logits": logits}

    @torch.no_grad()
    def __call__(self, image: np.ndarray) -> list[dict]:
        h, w = image.shape[:2]
        x = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        out = self.raw(normalize_crops(x[None], torch.device("cuda")))
        res = self.post(out, torch.tensor([[h, w]], device="cuda"))[0]
        scores, labels, boxes = (res[k].cpu().numpy() for k in ("scores", "labels", "boxes"))
        return [
            {
                "category_id": int(self.obj_ids[lbl]),
                "score": float(s),
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
            }
            for s, lbl, (x1, y1, x2, y2) in zip(scores, labels, boxes)
        ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default="small", choices=["tiny", "small", "medium"])
    parser.add_argument("--weights", required=True, help="best.pth of a bop_detector run")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--dataset", default="lmo", choices=sorted(SPLITS))
    parser.add_argument("--split", default="test")
    parser.add_argument("--img_size", type=int, default=640)
    parser.add_argument("--top_k", type=int, default=100)
    parser.add_argument("--fp32", action="store_true", help="fp32 engine instead of fp16")
    parser.add_argument("--check_images", type=int, default=8)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    os.makedirs(args.out_dir, exist_ok=True)
    tag = "fp32" if args.fp32 else "fp16"
    onnx_path = os.path.join(args.out_dir, f"lwdetr_{args.variant}.onnx")
    engine_path = os.path.join(args.out_dir, f"lwdetr_{args.variant}_{tag}.engine")

    obj_ids = SPLITS[args.dataset]["obj_ids"]
    model = BOPLWDETR(variant=args.variant, num_classes=len(obj_ids))
    model.load_state_dict(torch.load(args.weights, map_location="cpu", weights_only=True))
    model = model.cuda().eval()
    post = model._lwdetr_post["bbox"]
    ds = BOPDetDataset(load_index(args.dataset, args.split), args.img_size, augment=False)

    # Reference images: --check_images spread over the split, as the detector sees them
    picks = np.linspace(0, len(ds) - 1, args.check_images).astype(int)
    images = []
    for i in picks:
        img = ds.loader.load_image(ds.images[i][0])
        img = cv2.resize(img, (args.img_size, args.img_size), interpolation=cv2.INTER_LINEAR)
        images.append(normalize_crops(img[None], torch.device("cuda")))

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    net = export_model(model, args.img_size)
    with torch.no_grad():
        ref = [net(x) for x in images]  # (boxes, logits), fp32
    if not os.path.exists(onnx_path):
        logging.info(f"ONNX export -> {onnx_path}")
        torch.onnx.export(
            net, (images[0],), onnx_path, dynamo=True, opset_version=22, external_data=False,
            input_names=["images"], output_names=["pred_boxes", "pred_logits"],
        )
    if not os.path.exists(engine_path):
        logging.info(f"TensorRT build ({tag}) -> {engine_path}")
        build_engine(onnx_path, engine_path, fp16=not args.fp32, min_batch=1, opt_batch=1,
                     max_batch=1)

    detector = TRTDetector(engine_path, post, args.img_size, obj_ids, args.top_k)
    diff_box, diff_score = 0.0, 0.0
    with torch.no_grad():
        for x, (rb, rl) in zip(images, ref):
            out = detector.raw(x)
            torch.cuda.synchronize()
            diff_box = max(diff_box, (out["pred_boxes"] - rb).abs().max().item())
            diff_score = max(
                diff_score, (out["pred_logits"].sigmoid() - rl.sigmoid()).abs().max().item()
            )
    check = {"images": len(images), "max_abs_diff_boxes": diff_box,
             "max_abs_diff_scores": diff_score}
    logging.info(f"engine vs PyTorch fp32: {check}")

    out, times = detect_split(detector, ds)
    path = os.path.join(args.out_dir, f"lwdetr_trt_{tag}_{args.dataset}-{args.split}.json")
    with open(path, "w") as f:
        json.dump(out, f)
    summary = {
        "engine": os.path.abspath(engine_path), "weights": os.path.abspath(args.weights),
        "precision": tag, "check": check, "images": len(ds), "detections": len(out),
        "time_ms_median": 1000 * float(np.median(times)),
        "time_ms_mean": 1000 * float(np.mean(times)),
    }
    with open(path.replace(".json", "_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    logging.info(f"{path}: {summary}")


if __name__ == "__main__":
    main()
