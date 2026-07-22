"""Stage ① — face detection (SCRFD via InsightFace).

InsightFace ships SCRFD ONNX models and handles the (non-trivial) anchor decode
+ NMS for us. The ``buffalo_sc`` model pack uses **SCRFD-500M** — exactly the
detector recommended in CLAUDE.md — and auto-downloads on first use.

We only enable the ``detection`` module (bbox + 5 keypoints); recognition /
genderage are skipped to keep it light. Runs on GPU via onnxruntime's CUDA EP if
available, else CPU.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from .base import FaceBox, onnx_providers


class ScrfdFaceDetector:
    def __init__(
        self,
        model_pack: str = "buffalo_sc",   # SCRFD-500M detector
        det_size: int = 640,
        det_thresh: float = 0.5,
        use_gpu: bool = True,
    ):
        from insightface.app import FaceAnalysis  # lazy: heavy import

        providers = onnx_providers(use_gpu)
        gpu_active = "CUDAExecutionProvider" in providers
        self._app = FaceAnalysis(
            name=model_pack,
            allowed_modules=["detection"],
            providers=providers,
        )
        ctx_id = 0 if gpu_active else -1
        self._app.prepare(ctx_id=ctx_id, det_size=(det_size, det_size), det_thresh=det_thresh)

    def detect(self, frame_bgr: np.ndarray) -> List[FaceBox]:
        faces = self._app.get(frame_bgr)
        out: List[FaceBox] = []
        for f in faces:
            x0, y0, x1, y1 = f.bbox
            out.append(
                FaceBox(
                    x0=float(x0), y0=float(y0), x1=float(x1), y1=float(y1),
                    score=float(f.det_score),
                    kps=np.asarray(f.kps, dtype=np.float32) if f.kps is not None else None,
                )
            )
        return out

    def detect_largest(self, frame_bgr: np.ndarray) -> Optional[FaceBox]:
        """Cabin camera has one driver — return the biggest face, or None."""
        faces = self.detect(frame_bgr)
        if not faces:
            return None
        return max(faces, key=lambda b: b.width * b.height)
