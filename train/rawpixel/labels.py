"""Self-contained 3-class label derivation for the raw-pixel pipeline.

Deliberately duplicated from ``train/build_dataset.py`` rather than imported,
so this pipeline stays fully decoupled from the cascade/feature pipeline (see
docs/METHODOLOGY.md §14) — a change to one label-derivation copy can never
silently affect the other's results. The label RULES themselves are
intentionally identical (GT-PERCLOS 15s/20%, the same DISTRACTION_ACTIONS
set, DROWSY+TIRED collapsed into FATIGUED) so results stay comparable.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from ..dmd.annotations import FrameLabel

# driver_actions that mean the driver is distracted (not safe driving) —
# identical set to train/build_dataset.py::DISTRACTION_ACTIONS
DISTRACTION_ACTIONS = {
    "texting_left", "texting_right", "phonecall_left", "phonecall_right",
    "radio", "drinking", "hair_and_makeup", "reach_side", "reach_backseat",
    "talking_to_passenger", "change_gear",
}

THREE_STATES = ["FOCUSED", "DISTRACTED", "FATIGUED"]

DROWSY_PERCLOS = 0.20
DROWSY_WINDOW_S = 15.0


def _gt_perclos_rolling(eye_closed: np.ndarray, frame_idx: np.ndarray, fps: float,
                        window_s: float = DROWSY_WINDOW_S) -> np.ndarray:
    """Rolling fraction of GT eyes-closed over the last `window_s` seconds,
    time-indexed (not a fixed frame count) so it's correct regardless of
    stride — same approach as train/build_dataset.py::_gt_perclos."""
    s = pd.Series(eye_closed.astype(float))
    s.index = pd.to_timedelta(frame_idx / fps, unit="s")
    return s.rolling(f"{int(window_s)}s").mean().to_numpy()


def label_session(gt_labels: List[FrameLabel], task: str, frame_idx: np.ndarray,
                  fps: float) -> np.ndarray:
    """3-class label ('FOCUSED'/'DISTRACTED'/'FATIGUED'/None) for each frame in
    `frame_idx` (the native frame numbers actually extracted, e.g. after a
    face-detection gate — the GT-PERCLOS window is computed over exactly this
    subset, matching build_dataset.py's has_face-filter-then-roll order).

    `task` is 'drowsiness' or 'distraction' — DMD's two disjoint annotation
    protocols; each session's GT only fills the fields relevant to its task.
    """
    n = len(frame_idx)
    out = np.array([None] * n, dtype=object)
    if n == 0:
        return out

    if task == "drowsiness":
        eye_closed = np.array([
            1.0 if (i < len(gt_labels) and gt_labels[i].eye_closed == 1) else 0.0
            for i in frame_idx
        ])
        perclos = _gt_perclos_rolling(eye_closed, frame_idx, fps)
        for k, i in enumerate(frame_idx):
            if i >= len(gt_labels):
                continue
            lab = gt_labels[i]
            if lab.yawn == 1:
                out[k] = "FATIGUED"
            elif perclos[k] == perclos[k] and perclos[k] >= DROWSY_PERCLOS:  # not NaN
                out[k] = "FATIGUED"
            else:
                out[k] = "FOCUSED"
    elif task == "distraction":
        for k, i in enumerate(frame_idx):
            if i >= len(gt_labels):
                continue
            lab = gt_labels[i]
            act = lab.driver_action
            if lab.gaze_off_road == 1 or (isinstance(act, str) and act in DISTRACTION_ACTIONS):
                out[k] = "DISTRACTED"
            elif lab.looking_road is True and act == "safe_drive":
                out[k] = "FOCUSED"
            # else: unclassified/standstill/ambiguous -> left None, dropped downstream
    return out
