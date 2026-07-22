"""Stage ② — head pose (6DRepNet).

Replaces the prototype's ``cv2.solvePnP`` (whose yaw is broken, spans ±180°).
6DRepNet regresses full-range yaw/pitch/roll from a face crop via a continuous
6D rotation representation. The ``sixdrepnet`` pip package bundles the weights
and downloads them on first use.

We pass it a face crop (expanded around the SCRFD box) rather than the full
frame, so it never has to find the face itself.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .base import FaceBox, HeadPoseAngles


class SixDRepNetHeadPose:
    def __init__(self, use_gpu: bool = True, crop_margin: float = 0.25):
        import torch                          # lazy
        from sixdrepnet import SixDRepNet     # lazy: pulls torch weights

        gpu_id = 0 if (use_gpu and torch.cuda.is_available()) else -1
        self._model = SixDRepNet(gpu_id=gpu_id)
        self._margin = crop_margin

    def _crop(self, frame_bgr: np.ndarray, box: FaceBox) -> np.ndarray:
        h, w = frame_bgr.shape[:2]
        mx = box.width * self._margin
        my = box.height * self._margin
        x0 = max(0, int(box.x0 - mx))
        y0 = max(0, int(box.y0 - my))
        x1 = min(w, int(box.x1 + mx))
        y1 = min(h, int(box.y1 + my))
        return frame_bgr[y0:y1, x0:x1]

    def estimate(self, frame_bgr: np.ndarray, box: FaceBox) -> Optional[HeadPoseAngles]:
        crop = self._crop(frame_bgr, box)
        if crop.size == 0:
            return None
        # sixdrepnet returns (pitch, yaw, roll) in degrees
        pitch, yaw, roll = self._model.predict(crop)
        return HeadPoseAngles(
            yaw=float(np.ravel(yaw)[0]),
            pitch=float(np.ravel(pitch)[0]),
            roll=float(np.ravel(roll)[0]),
        )
