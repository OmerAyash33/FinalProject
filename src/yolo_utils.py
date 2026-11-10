"""Hardware-adaptive YOLO model loading: CUDA GPU -> raw PyTorch (FP16), else OpenVINO, else plain CPU."""
import os
import torch
from ultralytics import YOLO

from config import TARGET_WIDTH
from logging_utils import log


def load_yolo_model(cpu_model_path: str, gpu_model_path: str = None,
                     ov_dir: str = None, task: str = "detect"):
    """
    Three-tier hardware fallback, tried in order. Returns (model, use_fp16) -
    callers should pass `half=use_fp16` into their own .predict()/.track()
    calls, since that's a per-call argument, not something set on the model.

      1. CUDA GPU -> `gpu_model_path` (or `cpu_model_path` if no distinct
         GPU-sized weight exists) in raw PyTorch, with FP16 enabled: Tensor
         Core GPUs (RTX 20-series and newer) get roughly 2x throughput from
         half-precision inference, with no meaningful accuracy loss.
      2. No GPU -> `cpu_model_path` compiled to Intel OpenVINO (cached in
         `ov_dir`, auto-derived from the filename if not given) - the fast
         path for CPU inference. FP16 is not used here: OpenVINO's own IR
         precision handling already does the right thing per-device, and
         forcing half precision on plain CPU tensors is unsupported/slow.
      3. OpenVINO itself unavailable or fails to compile/load for any reason
         (not installed, unsupported CPU, etc.) -> `cpu_model_path` in plain
         PyTorch on CPU. Slowest tier, but runs on literally any machine.
    """
    if torch.cuda.is_available():
        chosen = gpu_model_path or cpu_model_path
        log(f"[INFO] CUDA GPU detected -> loading {chosen} on GPU (FP16).")
        return YOLO(chosen).to('cuda'), True

    if ov_dir is None:
        stem = os.path.splitext(os.path.basename(cpu_model_path))[0]
        ov_dir = os.path.join(os.path.dirname(cpu_model_path) or ".", f"{stem}_openvino_model")

    try:
        log("[INFO] No CUDA GPU detected -> trying CPU + Intel OpenVINO...")
        return export_and_load_openvino(cpu_model_path, ov_dir, task=task), False
    except Exception as exc:
        log(f"[WARNING] OpenVINO unavailable ({exc}) -> falling back to plain CPU (PyTorch).")
        return YOLO(cpu_model_path), False


def export_and_load_openvino(pt_path: str, ov_dir: str, task: str = "detect") -> YOLO:
    """
    Return a YOLO model running on the OpenVINO backend, which is drastically
    faster than raw PyTorch for CPU inference. Compiles `pt_path` to OpenVINO
    once (cached in `ov_dir`) and reuses that cache on every later call.
    """
    log("[INFO] Checking AI engine...")
    if not os.path.exists(ov_dir):
        # First run only: locks the compiled graph to TARGET_WIDTH, so this exact
        # export can't be reused if TARGET_WIDTH changes later without deleting ov_dir.
        log("[INFO] Compiling YOLO to OpenVINO...")
        base_model = YOLO(pt_path)
        base_model.export(format="openvino", imgsz=TARGET_WIDTH, dynamic=False)

    log("[INFO] Model loaded successfully.")
    return YOLO(ov_dir, task=task)
