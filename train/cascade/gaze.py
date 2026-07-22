"""Stage ⑤ — eye gaze (gaze-estimation-adas-0002, OpenVINO IR).

Optional finer-grained "where are the eyes pointing" signal. Where head pose
only knows where the *head* faces, this catches eyes-only glances (e.g. flicking
to the radio while the head stays forward) — the cases head pose alone misses for
DISTRACTED.

It composes neatly with the rest of the cascade: it consumes the two eye crops
(located from SCRFD's 5 keypoints) and the head-pose angles from 6DRepNet, and
outputs a 3D gaze vector which we turn into gaze yaw/pitch.

Model I/O (confirmed): left_eye_image[1,3,60,60], right_eye_image[1,3,60,60],
head_pose_angles[1,3] -> gaze_vector[1,3]. Eyes are expected as raw BGR 0-255.
"""
from __future__ import annotations

import math
import os
from typing import Optional

import numpy as np

from .base import FaceBox, HeadPoseAngles, GazeReading


class GazeEstimator:
    def __init__(self, model_xml: str, use_gpu: bool = True, eye_box_frac: float = 0.25):
        import openvino as ov  # lazy

        if not os.path.exists(model_xml):
            raise FileNotFoundError(
                f"Gaze IR not found: {model_xml}. Run `python -m train.download_models`."
            )
        core = ov.Core()
        model = core.read_model(model_xml)
        device = "GPU" if (use_gpu and "GPU" in core.available_devices) else "CPU"
        self._compiled = core.compile_model(model, device)
        self._out = self._compiled.output("gaze_vector")
        self._box_frac = eye_box_frac

    def _eye_crop(self, frame_bgr, center, half):
        import cv2
        h, w = frame_bgr.shape[:2]
        cx, cy = float(center[0]), float(center[1])
        x0 = max(0, int(cx - half)); x1 = min(w, int(cx + half))
        y0 = max(0, int(cy - half)); y1 = min(h, int(cy + half))
        crop = frame_bgr[y0:y1, x0:x1]
        if crop.size == 0:
            return None
        return cv2.resize(crop, (60, 60))

    @staticmethod
    def _prep(img):
        return np.transpose(img.astype(np.float32), (2, 0, 1))[None]  # NCHW, raw BGR

    def read(self, frame_bgr, box: FaceBox, pose: HeadPoseAngles) -> Optional[GazeReading]:
        if box.kps is None or len(box.kps) < 2 or pose is None:
            return None
        left_eye, right_eye = box.kps[0], box.kps[1]
        inter = float(np.linalg.norm(np.asarray(left_eye) - np.asarray(right_eye)))
        half = max(10.0, inter * self._box_frac)
        left = self._eye_crop(frame_bgr, left_eye, half)
        right = self._eye_crop(frame_bgr, right_eye, half)
        if left is None or right is None:
            return None

        # 6DRepNet's pitch sign is inverted vs the Intel head-pose convention the
        # gaze model was trained with; negate it so the head's vertical contribution
        # combines correctly (empirically: corr(head_pitch, gaze_y) 0.48 -> 0.69).
        hp = np.array([[pose.yaw, -pose.pitch, pose.roll]], dtype=np.float32)
        result = self._compiled({
            "left_eye_image": self._prep(left),
            "right_eye_image": self._prep(right),
            "head_pose_angles": hp,
        })
        gv = np.asarray(result[self._out]).ravel().astype(np.float64)
        n = np.linalg.norm(gv) + 1e-9
        gv = gv / n

        # The model emits gaze in an anatomical frame (subject's right = +x),
        # which on a driver-facing camera is image-LEFT. Flip x into image
        # coordinates so gaze_x>0 means image-right (matches the visible pupils).
        # Verified against known-direction frames; magnitude (validation) unchanged.
        gv = gv.copy()
        gv[0] = -gv[0]

        # Un-roll the gaze vector by head roll so yaw/pitch are in the upright
        # image frame (per the OMZ gaze demo).
        roll = math.radians(pose.roll)
        cs, sn = math.cos(roll), math.sin(roll)
        gx = gv[0] * cs + gv[1] * sn
        gy = -gv[0] * sn + gv[1] * cs
        gz = gv[2]
        # Model's forward axis is -z (looking ahead -> gz ~ -1), so deviation
        # from forward is atan2(component, -gz): forward -> 0 deg.
        fwd = -gz
        yaw = math.degrees(math.atan2(gx, fwd))
        pitch = math.degrees(math.atan2(gy, fwd))
        return GazeReading(x=float(gx), y=float(gy), z=float(gz),
                           yaw=float(yaw), pitch=float(pitch))
