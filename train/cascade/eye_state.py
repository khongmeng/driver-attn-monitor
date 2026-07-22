"""Stage ③ — eye open/closed (open-closed-eye-0001, ONNX).

Replaces the prototype's EAR threshold (the source of ~89% false DROWSY). The
OpenVINO ``open-closed-eye-0001`` model is a 32x32 classifier that is essentially
free (0.0014 GFLOPs, ~95.8% acc). We run it once per eye, using the SCRFD 5-point
keypoints to locate each eye, and feed the per-eye open-probability to the
temporal PERCLOS/blink layer.

The wrapper is generic: any 32x32-ish eye open/closed ONNX with a 2-logit (or
1-logit sigmoid) head works by tweaking ``input_size`` / ``open_index`` /
``bgr`` / ``scale``. See ``train/download_models.py`` for fetching the weights.
"""
from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np

from .base import FaceBox, EyeReading, onnx_providers


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / np.sum(e)


class OnnxEyeState:
    def __init__(
        self,
        onnx_path: str,
        input_size: int = 32,
        open_index: int = 1,      # output index for the "open" class
        bgr: bool = True,         # OMZ model expects BGR, raw 0-255
        scale: float = 1.0,       # multiply pixels by this after optional mean-sub
        mean: float = 0.0,
        eye_box_frac: float = 0.30,   # half-box size as fraction of inter-ocular dist
        use_gpu: bool = True,
    ):
        import onnxruntime as ort  # lazy

        if not os.path.exists(onnx_path):
            raise FileNotFoundError(
                f"Eye-state ONNX not found: {onnx_path}. "
                f"Run `python -m train.download_models` to fetch it."
            )
        self._sess = ort.InferenceSession(onnx_path, providers=onnx_providers(use_gpu))
        self._inp = self._sess.get_inputs()[0].name
        self._size = input_size
        self._open_index = open_index
        self._bgr = bgr
        self._scale = scale
        self._mean = mean
        self._box_frac = eye_box_frac

    # -- preprocessing -----------------------------------------------------
    def _prep(self, eye_bgr: np.ndarray) -> np.ndarray:
        import cv2
        img = cv2.resize(eye_bgr, (self._size, self._size))
        if not self._bgr:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = (img.astype(np.float32) - self._mean) * self._scale
        img = np.transpose(img, (2, 0, 1))[None]   # NCHW
        return img

    def _open_prob(self, eye_bgr: np.ndarray) -> float:
        if eye_bgr.size == 0:
            return float("nan")
        out = self._sess.run(None, {self._inp: self._prep(eye_bgr)})[0].ravel()
        if out.size == 1:                  # single-logit sigmoid head
            return float(1.0 / (1.0 + np.exp(-out[0])))
        # open-closed-eye-0001 already ends in softmax — use the outputs as
        # probabilities directly; only re-softmax if they're raw logits.
        if np.all(np.isfinite(out)) and np.all(out >= 0) and abs(out.sum() - 1.0) < 1e-3:
            probs = out
        else:
            probs = _softmax(out)
        return float(probs[self._open_index])

    # -- eye crops from 5-point keypoints ----------------------------------
    def _eye_crop(self, frame_bgr: np.ndarray, center, half: float) -> np.ndarray:
        h, w = frame_bgr.shape[:2]
        cx, cy = float(center[0]), float(center[1])
        x0 = max(0, int(cx - half)); x1 = min(w, int(cx + half))
        y0 = max(0, int(cy - half)); y1 = min(h, int(cy + half))
        return frame_bgr[y0:y1, x0:x1]

    def read(self, frame_bgr: np.ndarray, box: FaceBox) -> Optional[EyeReading]:
        if box.kps is None or len(box.kps) < 2:
            return None
        left_eye, right_eye = box.kps[0], box.kps[1]
        inter = float(np.linalg.norm(np.asarray(left_eye) - np.asarray(right_eye)))
        half = max(8.0, inter * self._box_frac)
        lp = self._open_prob(self._eye_crop(frame_bgr, left_eye, half))
        rp = self._open_prob(self._eye_crop(frame_bgr, right_eye, half))
        return EyeReading(left_open_prob=lp, right_open_prob=rp, source="model")
