"""Parse DMD drowsiness annotations (ASAM OpenLABEL JSON) into per-frame labels.

The DMD stores temporal *actions*, not per-frame labels. Each action has a
`type` (e.g. ``eyes_state/close``, ``blinks/blinking``, ``yawning/Yawning with
hand``) and a list of ``frame_intervals`` ``[{frame_start, frame_end}, ...]``
during which it is active. We invert that into one `FrameLabel` per frame so it
can be joined column-for-column against the features our cascade extracts.

Annotation levels (mutually exclusive within a level), per the DMD README:
  * eyes_state : open | close | opening | closing
  * blinks     : blinking
  * yawning    : "Yawning with hand" | "Yawning without hand"
  * occlusion  : face / body / hands  (kept as a flag; useful to drop frames)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# Which annotation "level" each action type prefix belongs to. The part before
# the "/" is the level; the part after is the label within that level.
# First four = drowsiness set; rest = distraction set.
ACTION_LEVELS = (
    "eyes_state", "blinks", "yawning", "occlusion",
    "gaze_on_road", "driver_actions", "talking", "hands_using_wheel",
)


@dataclass
class FrameLabel:
    """Ground truth for a single frame, flattened for CSV export.

    Drowsiness-set fields (eyes/blink/yawn) and distraction-set fields
    (gaze_on_road / driver_action / ...) coexist; a given session only fills the
    ones its annotation provides.
    """
    frame: int
    # Event bool fields are Optional: False = level annotated but not active this
    # frame, None = the session doesn't annotate that level at all (unlabeled).
    # --- drowsiness set ---
    eyes_state: Optional[str] = None      # open | close | opening | closing | None
    blink: Optional[bool] = False         # mid-blink per the blinks/blinking action
    yawn: Optional[bool] = False          # yawning (with or without hand)
    yawn_hand: Optional[bool] = False     # yawning *with* hand specifically
    occluded: Optional[bool] = False      # any occlusion level active
    # --- distraction set ---
    looking_road: Optional[bool] = None   # gaze_on_road: True=looking_road, False=not
    driver_action: Optional[str] = None   # safe_drive | radio | drinking | reach_side | ...
    talking: Optional[bool] = False       # talking/talking active
    hands_on_wheel: Optional[str] = None  # both | only_right | only_left

    @property
    def eye_closed(self) -> Optional[int]:
        """1 if eyes fully closed, 0 if open, None if transition/unlabeled.

        ``opening``/``closing`` are transitions — excluded so they don't muddy
        the open-vs-closed agreement check against the eye-state model.
        """
        if self.eyes_state == "close":
            return 1
        if self.eyes_state == "open":
            return 0
        return None

    @property
    def gaze_off_road(self) -> Optional[int]:
        """1 if not looking at road, 0 if looking, None if unlabeled."""
        if self.looking_road is None:
            return None
        return 0 if self.looking_road else 1


def _iter_actions(openlabel: dict):
    """Yield (action_type, [(start, end), ...]) for every annotated action."""
    actions = openlabel.get("actions", {}) or {}
    for action in actions.values():
        atype = action.get("type")
        if not atype:
            continue
        intervals = []
        for iv in action.get("frame_intervals", []) or []:
            start = iv.get("frame_start")
            end = iv.get("frame_end")
            if start is None or end is None:
                continue
            intervals.append((int(start), int(end)))
        if intervals:
            yield atype, intervals


def _frame_count(openlabel: dict, actions: list) -> int:
    """Best-effort total frame count: explicit frames map, else max interval+1."""
    frames = openlabel.get("frames")
    if isinstance(frames, dict) and frames:
        # frame keys are stringified ints
        return max(int(k) for k in frames.keys()) + 1
    max_end = 0
    for _atype, intervals in actions:
        for _start, end in intervals:
            max_end = max(max_end, end)
    return max_end + 1


def load_frame_labels(json_path: str, n_frames: Optional[int] = None) -> List[FrameLabel]:
    """Load a DMD annotation JSON and return one `FrameLabel` per frame.

    Args:
        json_path: path to a ``*_ann_drowsiness.json`` file.
        n_frames:  if given, force the table to this length (e.g. the actual
                   video frame count). Frames beyond the annotations stay blank;
                   annotations beyond ``n_frames`` are ignored.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    openlabel = data.get("openlabel", data)

    actions = list(_iter_actions(openlabel))
    total = n_frames if n_frames is not None else _frame_count(openlabel, actions)
    labels = [FrameLabel(frame=i) for i in range(total)]

    # Which annotation levels this file actually contains. Event-bool fields for
    # levels the file does NOT annotate are set to None (unlabeled) rather than
    # False, so cross-set training never treats an un-annotated level as "0".
    present = {atype.split("/", 1)[0] for atype, _ in actions}
    for lab in labels:
        if "blinks" not in present:
            lab.blink = None
        if "yawning" not in present:
            lab.yawn = None
            lab.yawn_hand = None
        if "occlusion" not in present:
            lab.occluded = None
        if "talking" not in present:
            lab.talking = None

    for atype, intervals in actions:
        level, _, label = atype.partition("/")
        for start, end in intervals:
            for fr in range(start, min(end, total - 1) + 1):
                lab = labels[fr]
                if level == "eyes_state":
                    lab.eyes_state = label
                elif level == "blinks":
                    lab.blink = True
                elif level == "yawning":
                    lab.yawn = True
                    if "with hand" in label.lower():
                        lab.yawn_hand = True
                elif level == "occlusion":
                    lab.occluded = True
                elif level == "gaze_on_road":
                    lab.looking_road = (label == "looking_road")
                elif level == "driver_actions":
                    lab.driver_action = label
                elif level == "talking":
                    lab.talking = True
                elif level == "hands_using_wheel":
                    lab.hands_on_wheel = label
    return labels


def summarize(labels: List[FrameLabel]) -> Dict[str, int]:
    """Quick counts for sanity-checking a parse."""
    out = {
        "frames": len(labels),
        "eyes_open": sum(1 for l in labels if l.eyes_state == "open"),
        "eyes_close": sum(1 for l in labels if l.eyes_state == "close"),
        "eyes_transition": sum(1 for l in labels if l.eyes_state in ("opening", "closing")),
        "blink": sum(1 for l in labels if l.blink),
        "yawn": sum(1 for l in labels if l.yawn),
        "occluded": sum(1 for l in labels if l.occluded),
        "looking_road": sum(1 for l in labels if l.looking_road is True),
        "not_looking_road": sum(1 for l in labels if l.looking_road is False),
    }
    return out
