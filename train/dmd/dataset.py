"""Discover DMD recording sessions on disk and pair videos with annotations.

Expected (extracted) layout, e.g. for drowsiness:

    datasets/DMD/drowsiness/dmd-dataset-drowsiness-gA-1/
        dmd/gA/1/s5/
            gA_1_s5_<ts>_rgb_face.mp4      <- driver-facing RGB (what we infer on)
            gA_1_s5_<ts>_rgb_body.mp4
            gA_1_s5_<ts>_rgb_mosaic.avi
            gA_1_s5_<ts>_rgb_ann_drowsiness.json   <- OpenLABEL ground truth

We key off the ``*_rgb_face.mp4`` videos and look for a sibling ``*_ann_*.json``.
Sessions still packed in ``.tar.gz`` are ignored (extract them first).
"""
from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass
from typing import List, Optional

# group / subject / session pulled from the canonical DMD file stem
#   e.g. "gA_1_s5_2019-03-14T14;26;17+01;00_rgb_face"
_STEM_RE = re.compile(r"^(?P<group>g\w+)_(?P<subject>\d+)_(?P<session>s\d+)_")


@dataclass
class DmdSession:
    face_video: str                  # absolute path to *_rgb_face.mp4
    annotation: Optional[str]        # absolute path to *_ann_*.json (or None)
    task: str                        # drowsiness | distraction | gaze | unknown
    group: str = ""                  # gA, gB, ...
    subject: str = ""                # participant id
    session: str = ""                # s5, ...

    @property
    def name(self) -> str:
        base = os.path.basename(self.face_video)
        return base.replace("_rgb_face.mp4", "")

    @property
    def has_annotation(self) -> bool:
        return self.annotation is not None


def _task_from_path(path: str) -> str:
    low = path.lower()
    for task in ("drowsiness", "distraction", "gaze"):
        if task in low:
            return task
    return "unknown"


def _find_annotation(face_video: str) -> Optional[str]:
    """Find the OpenLABEL JSON sitting next to a face video."""
    folder = os.path.dirname(face_video)
    stem = os.path.basename(face_video).replace("_rgb_face.mp4", "")
    # canonical name: <stem>_rgb_ann_<task>.json
    candidates = glob.glob(os.path.join(folder, f"{stem}_rgb_ann_*.json"))
    if candidates:
        return candidates[0]
    # fall back to any annotation json in the same session folder
    candidates = glob.glob(os.path.join(folder, "*_ann_*.json"))
    return candidates[0] if candidates else None


def find_sessions(
    dmd_root: str = "datasets/DMD",
    task: Optional[str] = None,
    require_annotation: bool = True,
) -> List[DmdSession]:
    """Recursively find extracted DMD sessions under ``dmd_root``.

    Args:
        dmd_root:           root holding ``drowsiness/``, ``distraction/`` ...
        task:               restrict to one task (e.g. ``"drowsiness"``).
        require_annotation: drop sessions with no annotation JSON.
    """
    pattern = os.path.join(dmd_root, "**", "*_rgb_face.mp4")
    sessions: List[DmdSession] = []
    for face in sorted(glob.glob(pattern, recursive=True)):
        face = os.path.abspath(face)
        sess_task = _task_from_path(face)
        if task and sess_task != task:
            continue
        ann = _find_annotation(face)
        if require_annotation and ann is None:
            continue
        meta = _STEM_RE.match(os.path.basename(face))
        sessions.append(
            DmdSession(
                face_video=face,
                annotation=ann,
                task=sess_task,
                group=meta.group("group") if meta else "",
                subject=meta.group("subject") if meta else "",
                session=meta.group("session") if meta else "",
            )
        )
    return sessions
