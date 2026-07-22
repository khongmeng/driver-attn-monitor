"""Production feature-extraction cascade (PC / offline).

A chain of pretrained models, each replacing one fragile geometric heuristic in
the MediaPipe prototype (see project CLAUDE.md):

    frame ─▶ ① SCRFD face detect ─▶ crop
                                     ├─▶ ② 6DRepNet head pose  (yaw/pitch/roll)
                                     └─▶ ③ eye-state model      (open/closed)
                                            └─▶ temporal: PERCLOS, blink rate

Each stage is a small swappable wrapper with a common interface, so a stage can
be replaced (or stubbed) without touching the rest. The orchestrator
`CascadePipeline` runs the whole chain on one frame and returns a flat
`FrameFeatures` row ready for CSV export and, later, training the Stage-④ state
classifier.

Heavy deps (cv2 / onnxruntime / torch / insightface / sixdrepnet) are imported
lazily inside each wrapper's ``__init__`` so importing this package is cheap and
a missing optional dep only breaks the stage that needs it.
"""

# Make torch's bundled CUDA 12 + cuDNN 9 DLLs discoverable by onnxruntime-gpu and
# InsightFace on Windows — otherwise the CUDA provider fails to load
# (cublasLt64_12.dll / cudnn64_9.dll not found) and silently falls back to CPU.
# Must run before any onnxruntime session is created.
import os as _os
try:
    import torch as _torch
    _torch_lib = _os.path.join(_os.path.dirname(_torch.__file__), "lib")
    if _os.path.isdir(_torch_lib) and hasattr(_os, "add_dll_directory"):
        _os.add_dll_directory(_torch_lib)
except Exception:  # noqa: BLE001 — GPU is optional; CPU still works
    pass

from .base import FaceBox, HeadPoseAngles, EyeReading, FrameFeatures
from .pipeline import CascadePipeline, build_pipeline

__all__ = [
    "FaceBox",
    "HeadPoseAngles",
    "EyeReading",
    "FrameFeatures",
    "CascadePipeline",
    "build_pipeline",
]
