"""DMD (Driver Monitoring Dataset) utilities for the PC training pipeline.

Two responsibilities:
  * `annotations` — turn a DMD OpenLABEL `*_ann_*.json` into a per-frame
    ground-truth table (eyes-state / blink / yawn).
  * `dataset`     — walk `datasets/DMD/...` and pair each RGB face video with
    its annotation JSON.

Nothing here imports cv2 / torch / onnxruntime, so it stays cheap to import and
easy to unit-test.
"""

from .annotations import FrameLabel, load_frame_labels, ACTION_LEVELS
from .dataset import DmdSession, find_sessions

__all__ = [
    "FrameLabel",
    "load_frame_labels",
    "ACTION_LEVELS",
    "DmdSession",
    "find_sessions",
]
