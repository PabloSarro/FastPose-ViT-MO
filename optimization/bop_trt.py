# M9: TensorRT engine for the BOP pose network exported by optimization/bop_export.py, and a
# drop-in runner that src/bop_evaluate.py uses in place of the PyTorch model (--backend trt).
#
# The engine keeps the ONNX signature: images (N, 3, S, S) float32, obj_idx (N,) int64 ->
# pose (N, 9) float32. With --fp16, TensorRT may run layers in half precision inside; inputs
# and outputs stay float32. One optimization profile covers N = --min_batch..--max_batch,
# tuned for --opt_batch (default 1 / 8 / 16 crops, M9 bullet 2).
#
# Build (from the repo root):
#   PYTHONPATH=. .venv/bin/python optimization/bop_trt.py --onnx_path OUT/model.onnx \
#       --engine_path OUT/model_fp16.engine --fp16
# With --model_weights (and --dataset/--object_conditioning), the engine is also compared with
# PyTorch fp32 (TF32 off) on real crops, as in bop_export.py: report in <engine>.check.json.

import argparse
import json
import logging
import os
import time

import tensorrt as trt
import torch

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def build_engine(
    onnx_path: str,
    engine_path: str,
    fp16: bool,
    min_batch: int = 1,
    opt_batch: int = 8,
    max_batch: int = 16,
    workspace_mb: int = 4096,
) -> dict:
    """Build and save a TensorRT engine with one dynamic-batch profile. Returns build info."""
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(0)  # explicit batch is the only mode in TensorRT 10
    parser = trt.OnnxParser(network, TRT_LOGGER)
    if not parser.parse_from_file(onnx_path):
        errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise RuntimeError(f"ONNX parsing failed: {errors}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb << 20)
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    else:
        # TensorRT enables TF32 by default (max |diff| 4e-3 vs PyTorch): the fp32 engine is the
        # strict reference, so turn it off
        config.clear_flag(trt.BuilderFlag.TF32)
    profile = builder.create_optimization_profile()
    for i in range(network.num_inputs):
        inp = network.get_input(i)
        rest = tuple(inp.shape)[1:]
        profile.set_shape(
            inp.name, (min_batch, *rest), (opt_batch, *rest), (max_batch, *rest)
        )
    config.add_optimization_profile(profile)

    start = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT engine build failed")
    os.makedirs(os.path.dirname(os.path.abspath(engine_path)), exist_ok=True)
    with open(engine_path, "wb") as f:
        f.write(serialized)
    info = {
        "onnx_path": os.path.abspath(onnx_path),
        "engine_path": os.path.abspath(engine_path),
        "fp16": fp16,
        "profile": [min_batch, opt_batch, max_batch],
        "build_seconds": round(time.perf_counter() - start, 1),
        "engine_mb": round(os.path.getsize(engine_path) / 2**20, 1),
        "tensorrt": trt.__version__,
        "gpu": torch.cuda.get_device_name(),
    }
    with open(engine_path + ".json", "w") as f:
        json.dump(info, f, indent=2)
    return info


class TRTPoseModel:
    """Runs a pose engine like FastPoseViT: model(images, obj_idx) -> pose (N, 9) on the GPU.

    Inputs are torch CUDA tensors. The engine runs on its own stream (TensorRT adds syncs on
    the default stream), ordered after and before torch's current stream (wait_stream), so
    callers see it like any torch op and torch.cuda.synchronize() timing works.
    """

    def __init__(self, engine_path: str) -> None:
        self.runtime = trt.Runtime(TRT_LOGGER)
        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"cannot load TensorRT engine {engine_path}")
        self.context = self.engine.create_execution_context()
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        modes = {n: self.engine.get_tensor_mode(n) for n in names}
        self.inputs = [n for n in names if modes[n] == trt.TensorIOMode.INPUT]
        self.outputs = [n for n in names if modes[n] == trt.TensorIOMode.OUTPUT]
        if self.inputs != ["images", "obj_idx"] or self.outputs != ["pose"]:
            raise RuntimeError(f"unexpected engine I/O {self.inputs} -> {self.outputs}")
        _, _, self.max_batch = self.engine.get_tensor_profile_shape("images", 0)
        self.max_batch = self.max_batch[0]
        self.device = torch.device("cuda")
        self.stream = torch.cuda.Stream()

    # Like nn.Module, so that bop_evaluate.py can call .eval() and .parameters()-free helpers
    def eval(self):
        return self

    def __call__(self, images: torch.Tensor, obj_idx: torch.Tensor) -> torch.Tensor:
        if len(images) > self.max_batch:  # larger batches in chunks of the profile's maximum
            return torch.cat(
                [
                    self(images[i : i + self.max_batch], obj_idx[i : i + self.max_batch])
                    for i in range(0, len(images), self.max_batch)
                ]
            )
        images = images.to(self.device, torch.float32).contiguous()
        obj_idx = obj_idx.to(self.device, torch.int64).contiguous()
        out = torch.empty(len(images), 9, device=self.device, dtype=torch.float32)
        self.context.set_input_shape("images", tuple(images.shape))
        self.context.set_input_shape("obj_idx", tuple(obj_idx.shape))
        self.context.set_tensor_address("images", images.data_ptr())
        self.context.set_tensor_address("obj_idx", obj_idx.data_ptr())
        self.context.set_tensor_address("pose", out.data_ptr())
        caller = torch.cuda.current_stream()
        self.stream.wait_stream(caller)  # inputs ready
        if not self.context.execute_async_v3(self.stream.cuda_stream):
            raise RuntimeError("TensorRT execution failed")
        caller.wait_stream(self.stream)  # output ready for the caller's next ops
        # The tensors were made on the caller's stream and used on ours: keep them alive
        for t in (images, obj_idx, out):
            t.record_stream(self.stream)
        return out


def compare(engine_path: str, args: argparse.Namespace) -> dict:
    """Max |engine - PyTorch fp32| on real crops, per batch size (cf. bop_export.check)."""
    import numpy as np

    from optimization.bop_export import real_crops
    from src.bop_evaluate import build_model
    from src.models import VIT_MODELS

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = build_model(args, torch.device("cuda"))
    images, obj_idx = real_crops(
        args.dataset, "test", args.check_crops, VIT_MODELS[args.vit_model][3][0], args.depth_mode
    )
    images, obj_idx = images.cuda(), obj_idx.cuda()
    trt_model = TRTPoseModel(engine_path)
    with torch.no_grad():
        ref = model(images, obj_idx).float()
    report = {"n_crops": len(images), "results": {}}
    for bs in args.batch_sizes:
        out = torch.cat(
            [trt_model(images[i : i + bs], obj_idx[i : i + bs]) for i in range(0, len(images), bs)]
        )
        diff = (out - ref).abs()
        # Per part: translation (3 values) and 6D rotation (6 values)
        report["results"][str(bs)] = {
            "max_abs_diff": diff.max().item(),
            "mean_abs_diff": diff.mean().item(),
            "max_abs_diff_t": diff[:, :3].max().item(),
            "max_abs_diff_R": diff[:, 3:].max().item(),
        }
        logging.info(f"batch {bs}: {report['results'][str(bs)]}")
    report["max_abs_diff"] = float(np.max([r["max_abs_diff"] for r in report["results"].values()]))
    with open(engine_path + ".check.json", "w") as f:
        json.dump(report, f, indent=2)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a TensorRT engine for the BOP pose model")
    parser.add_argument("--onnx_path", required=True)
    parser.add_argument("--engine_path", required=True)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--min_batch", type=int, default=1)
    parser.add_argument("--opt_batch", type=int, default=8)
    parser.add_argument("--max_batch", type=int, default=16)
    parser.add_argument("--workspace_mb", type=int, default=4096)
    parser.add_argument("--skip_build", action="store_true", help="only compare an existing engine")
    # Comparison with PyTorch (optional)
    parser.add_argument("--model_weights", default=None)
    parser.add_argument("--dataset", default="lmo")
    parser.add_argument("--vit_model", default="vit_b_16_384")
    parser.add_argument("--depth_mode", default="site")
    parser.add_argument("--object_conditioning", default="token")
    parser.add_argument("--check_crops", type=int, default=48)
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=[1, 4, 8, 16])
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    if not args.skip_build:
        info = build_engine(
            args.onnx_path,
            args.engine_path,
            args.fp16,
            args.min_batch,
            args.opt_batch,
            args.max_batch,
            args.workspace_mb,
        )
        logging.info(json.dumps(info))
    if args.model_weights:
        report = compare(args.engine_path, args)
        logging.info(f"max |engine - PyTorch fp32| = {report['max_abs_diff']:.2e}")


if __name__ == "__main__":
    main()
