"""Shared dataclasses and interfaces for the cascade stages."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple

import numpy as np


def onnx_providers(use_gpu: bool = True) -> List[str]:
    """ONNX Runtime providers, filtered to those actually installed.

    Requesting ``CUDAExecutionProvider`` with the CPU-only onnxruntime build
    just prints a warning; intersecting with the available set keeps logs clean
    and lets the same config run on a CPU box or a GPU box unchanged.
    """
    try:
        import onnxruntime as ort
        available = set(ort.get_available_providers())
    except Exception:  # noqa: BLE001
        return ["CPUExecutionProvider"]
    wanted = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_gpu else ["CPUExecutionProvider"]
    providers = [p for p in wanted if p in available]
    return providers or ["CPUExecutionProvider"]


@dataclass
class FaceBox:
    """Output of stage ① (face detection)."""
    x0: float
    y0: float
    x1: float
    y1: float
    score: float
    # 5-point keypoints (left_eye, right_eye, nose, left_mouth, right_mouth) if
    # the detector provides them — used to crop eyes for stage ③.
    kps: Optional[np.ndarray] = None

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    def as_int(self) -> Tuple[int, int, int, int]:
        return int(self.x0), int(self.y0), int(self.x1), int(self.y1)


@dataclass
class HeadPoseAngles:
    """Output of stage ② (head pose), degrees."""
    yaw: float
    pitch: float
    roll: float


@dataclass
class EyeReading:
    """Output of stage ③ (eye state)."""
    left_open_prob: float
    right_open_prob: float
    source: str = "model"          # "model" (ONNX) or "ear" (landmark fallback)

    @property
    def open_prob(self) -> float:
        return 0.5 * (self.left_open_prob + self.right_open_prob)


@dataclass
class GazeReading:
    """Output of stage ⑤ (eye gaze)."""
    x: float                       # normalized gaze vector (right/up/forward)
    y: float
    z: float
    yaw: float                     # degrees, horizontal gaze angle
    pitch: float                   # degrees, vertical gaze angle

    @property
    def magnitude(self) -> float:
        return float((self.yaw ** 2 + self.pitch ** 2) ** 0.5)


@dataclass
class FrameFeatures:
    """One flat row of extracted features for a single frame (CSV-ready)."""
    frame: int
    has_face: int = 0                 # 0/1 — drives the NO_FACE gate
    det_score: float = 0.0
    bbox_x0: float = 0.0
    bbox_y0: float = 0.0
    bbox_x1: float = 0.0
    bbox_y1: float = 0.0
    # stage ② head pose
    yaw: float = float("nan")
    pitch: float = float("nan")
    roll: float = float("nan")
    # stage ③ eye state
    left_open_prob: float = float("nan")
    right_open_prob: float = float("nan")
    eye_open_prob: float = float("nan")
    eye_closed: int = 0               # 0/1 thresholded
    eye_source: str = ""              # "model" | "ear"
    # stage ⑤ eye gaze
    gaze_yaw: float = float("nan")
    gaze_pitch: float = float("nan")
    gaze_x: float = float("nan")
    gaze_y: float = float("nan")
    gaze_z: float = float("nan")
    # temporal (filled by the aggregator)
    perclos: float = float("nan")
    blink: int = 0                    # blink completed on this frame
    blink_count: int = 0              # cumulative
    blink_rate: float = float("nan")  # blinks/minute
    # stage — mouth (MAR / yawn)
    mar: float = float("nan")         # mouth aspect ratio (opening / width)
    mouth_open: int = 0               # 0/1 thresholded
    yawn_active: int = 0              # sustained mouth-open run has crossed the min duration
    yawn_count: int = 0               # cumulative
    yawn_rate: float = float("nan")   # yawns/minute

    def to_dict(self) -> dict:
        return asdict(self)
