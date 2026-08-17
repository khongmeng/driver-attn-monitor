"""Stage — mouth landmarks -> Mouth Aspect Ratio (MAR), for yawn detection.

No pretrained mouth-open/closed ONNX model exists (unlike open-closed-eye-0001
for eyes — checked; see docs/METHODOLOGY.md §8.7). The established approach
(the ``OpenVino-Driver-Behaviour`` reference project already cited in
CLAUDE.md, and standard in the yawn-detection literature) is a dense
facial-landmark model -> Mouth Aspect Ratio -> threshold/temporal-integrate
for yawn, the same ratio-of-distances structure as EAR. Uses InsightFace's
``2d106det`` (106-point 2D landmarks, from the ``buffalo_s`` pack —
``buffalo_sc``, used for stage ① detection, has no landmark model bundled).

**Landmark indices were verified empirically** against real driver frames
(not just assumed from documentation) — see docs/METHODOLOGY.md §8.7: points
52-71 are the 20-point mouth cluster, and 52 / 61 are the horizontal extremes
(mouth corners) in every test frame checked. The commonly-documented
inner/outer lip split (52-61 outer, 62-71 inner) could not be confirmed as
cleanly on the sample frames available (one had only a partial mouth opening,
another had a hand occluding the mouth mid-yawn), so MAR uses the full
20-point mouth cluster's vertical span rather than betting on a specific
inner-vs-outer subset — both contours separate together when the mouth
opens, so the signal survives either way.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .base import FaceBox, onnx_providers

MOUTH_START, MOUTH_END = 52, 72   # indices [52, 71] inclusive, verified empirically
LEFT_CORNER, RIGHT_CORNER = 52, 61  # horizontal extremes, verified empirically


@dataclass
class MouthReading:
    mar: float          # mouth aspect ratio: vertical span / mouth width
    mouth_open: int      # 0/1, thresholded


class Landmark106Mouth:
    def __init__(self, onnx_path: str, open_thresh: float = 0.5, use_gpu: bool = True):
        import onnxruntime as ort
        from insightface.model_zoo.landmark import Landmark

        if not os.path.exists(onnx_path):
            raise FileNotFoundError(
                f"Landmark ONNX not found: {onnx_path}. "
                f"Run `python -m train.download_models` to fetch it."
            )
        providers = onnx_providers(use_gpu)
        session = ort.InferenceSession(onnx_path, providers=providers)
        self._lmk = Landmark(model_file=onnx_path, session=session)
        self._lmk.prepare(ctx_id=0 if "CUDAExecutionProvider" in session.get_providers() else -1)
        self._thresh = open_thresh

    def read(self, frame_bgr: np.ndarray, box: FaceBox) -> Optional[MouthReading]:
        face = _FaceShim(box)
        pts = self._lmk.get(frame_bgr, face)
        mouth = pts[MOUTH_START:MOUTH_END]
        width = float(abs(mouth[RIGHT_CORNER - MOUTH_START][0] - mouth[LEFT_CORNER - MOUTH_START][0]))
        # Guard against a degenerate landmark fit (both mouth corners collapsing
        # to nearly the same x) rather than a real narrow mouth — a real mouth
        # in these face crops is always tens of pixels wide; found via §8.7's
        # feature audit (a width of 0.024px produced mar=552.67, another
        # 0.28px -> mar=52.65 — the same divide-by-near-zero shape as the
        # blink_rate bug in §6.1). 1e-3 was far too permissive to catch this.
        if width < 10.0:
            return None
        opening = float(mouth[:, 1].max() - mouth[:, 1].min())
        mar = opening / width
        return MouthReading(mar=mar, mouth_open=int(mar > self._thresh))


class _FaceShim(dict):
    """Minimal duck-typed stand-in for insightface's Face object. `Landmark.get`
    reads `.bbox` and also writes its result back via `face[taskname] = pred`
    (insightface's real Face is dict-like) — subclassing dict gives us that
    item-assignment support for free, alongside the `.bbox` attribute."""
    def __init__(self, box: FaceBox):
        super().__init__()
        self.bbox = np.array([box.x0, box.y0, box.x1, box.y1], dtype=np.float32)
