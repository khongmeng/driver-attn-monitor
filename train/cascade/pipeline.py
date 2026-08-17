"""Cascade orchestrator — runs all stages on a frame and returns FrameFeatures.

`build_pipeline(cfg)` constructs the stages from a config dict (the ``cascade``
section of ``config.yaml``). Stages are optional: if the eye-state ONNX is
missing the pipeline still runs face + head pose (eye columns stay NaN), so you
can validate the parts you have and add the rest later.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .base import FrameFeatures
from .temporal import TemporalAggregator


class CascadePipeline:
    def __init__(self, detector, head_pose=None, eye_state=None, gaze=None, mouth=None, temporal=None):
        self.detector = detector
        self.head_pose = head_pose
        self.eye_state = eye_state
        self.gaze = gaze
        self.mouth = mouth
        self.temporal = temporal or TemporalAggregator()

    def reset(self):
        """Clear temporal state between sessions/videos — rebuilds with every
        constructor param the current aggregator has (not just 3 of them),
        so config-provided overrides survive past the first session instead
        of silently reverting to the dataclass defaults."""
        t = self.temporal
        self.temporal = TemporalAggregator(
            t.eye_thresh, t.blink_min, t.window, t.blink_rate_min_elapsed, t.yawn_min_duration
        )

    def process_frame(self, frame_bgr: np.ndarray, frame_idx: int, ts: float) -> FrameFeatures:
        feat = FrameFeatures(frame=frame_idx)

        box = self.detector.detect_largest(frame_bgr)
        if box is None:
            # No face: still advance the temporal clock with an "open" reading so
            # PERCLOS windows stay time-accurate, but flag NO_FACE via has_face=0.
            feat.has_face = 0
            return feat

        feat.has_face = 1
        feat.det_score = box.score
        feat.bbox_x0, feat.bbox_y0, feat.bbox_x1, feat.bbox_y1 = (
            box.x0, box.y0, box.x1, box.y1
        )

        # ② head pose
        pose = None
        if self.head_pose is not None:
            pose = self.head_pose.estimate(frame_bgr, box)
            if pose is not None:
                # Store head pose in the same image frame as gaze so the two read
                # consistently (+yaw = image-right, -pitch = down). 6DRepNet's yaw
                # is mirrored vs the image (its +yaw = image-left), so negate it;
                # its pitch already matches. The gaze stage below still receives
                # the raw `pose` object it was calibrated against.
                feat.yaw = -pose.yaw
                feat.pitch = pose.pitch
                feat.roll = pose.roll

        # ⑤ eye gaze (needs head pose + eye crops)
        if self.gaze is not None and pose is not None:
            g = self.gaze.read(frame_bgr, box, pose)
            if g is not None:
                feat.gaze_yaw, feat.gaze_pitch = g.yaw, g.pitch
                feat.gaze_x, feat.gaze_y, feat.gaze_z = g.x, g.y, g.z

        # ③ eye state
        eye_open_prob = float("nan")
        if self.eye_state is not None:
            reading = self.eye_state.read(frame_bgr, box)
            if reading is not None:
                feat.left_open_prob = reading.left_open_prob
                feat.right_open_prob = reading.right_open_prob
                eye_open_prob = reading.open_prob
                feat.eye_open_prob = eye_open_prob
                feat.eye_source = reading.source

        # temporal: PERCLOS / blink (driven by eye-state model)
        closed, blink_now, perclos, blink_rate = self.temporal.update(eye_open_prob, ts)
        feat.eye_closed = closed
        feat.blink = blink_now
        feat.blink_count = self.temporal.blink_count
        feat.blink_rate = blink_rate
        feat.perclos = perclos

        # mouth (MAR -> yawn), independent of the eye/gaze stages above.
        # Only advances the yawn run-length clock if the stage is actually
        # enabled — otherwise yawn_count/yawn_rate stay at their FrameFeatures
        # defaults (0 / NaN) rather than misleadingly reporting "zero yawns
        # measured" for a stage that never ran.
        if self.mouth is not None:
            reading = self.mouth.read(frame_bgr, box)
            mouth_open = False
            if reading is not None:
                feat.mar = reading.mar
                feat.mouth_open = reading.mouth_open
                mouth_open = bool(reading.mouth_open)
            yawn_active, yawn_count, yawn_rate = self.temporal.update_mouth(mouth_open, ts)
            feat.yawn_active = yawn_active
            feat.yawn_count = yawn_count
            feat.yawn_rate = yawn_rate
        return feat


def build_pipeline(cfg: dict) -> CascadePipeline:
    """Construct a CascadePipeline from the ``cascade`` config section.

    Missing/disabled stages degrade gracefully with a printed warning.
    """
    cfg = cfg or {}
    use_gpu = cfg.get("use_gpu", True)

    # ① face detection (required)
    from .face_detector import ScrfdFaceDetector
    fc = cfg.get("face", {})
    detector = ScrfdFaceDetector(
        model_pack=fc.get("model_pack", "buffalo_sc"),
        det_size=fc.get("det_size", 640),
        det_thresh=fc.get("det_thresh", 0.5),
        use_gpu=use_gpu,
    )

    # ② head pose (optional but recommended)
    head_pose = None
    if cfg.get("head_pose", {}).get("enabled", True):
        try:
            from .head_pose import SixDRepNetHeadPose
            head_pose = SixDRepNetHeadPose(
                use_gpu=use_gpu,
                crop_margin=cfg.get("head_pose", {}).get("crop_margin", 0.25),
            )
        except Exception as e:   # noqa: BLE001 — keep pipeline usable
            print(f"[cascade] head-pose stage disabled: {e}")

    # ③ eye state (optional — needs the ONNX weights)
    eye_state = None
    ec = cfg.get("eye_state", {})
    if ec.get("enabled", True):
        try:
            from .eye_state import OnnxEyeState
            eye_state = OnnxEyeState(
                onnx_path=ec.get("onnx_path", "models/eye_state/open_closed_eye.onnx"),
                input_size=ec.get("input_size", 32),
                open_index=ec.get("open_index", 1),
                bgr=ec.get("bgr", True),
                scale=ec.get("scale", 1.0),
                mean=ec.get("mean", 0.0),
                eye_box_frac=ec.get("eye_box_frac", 0.30),
                use_gpu=use_gpu,
            )
        except Exception as e:   # noqa: BLE001
            print(f"[cascade] eye-state stage disabled: {e}")

    # ⑤ eye gaze (optional — needs OpenVINO + the gaze IR; off by default)
    gaze = None
    gc = cfg.get("gaze", {})
    if gc.get("enabled", False):
        try:
            from .gaze import GazeEstimator
            gaze = GazeEstimator(
                model_xml=gc.get("model_xml", "models/gaze/gaze-estimation-adas-0002.xml"),
                use_gpu=use_gpu,
                eye_box_frac=gc.get("eye_box_frac", 0.5),
            )
        except Exception as e:   # noqa: BLE001
            print(f"[cascade] gaze stage disabled: {e}")

    # mouth / yawn (optional — needs the 2d106det landmark ONNX)
    mouth = None
    mc = cfg.get("mouth", {})
    if mc.get("enabled", True):
        try:
            from .mouth import Landmark106Mouth
            mouth = Landmark106Mouth(
                onnx_path=mc.get("onnx_path", "models/landmark/2d106det.onnx"),
                open_thresh=mc.get("open_thresh", 0.64),
                use_gpu=use_gpu,
            )
        except Exception as e:   # noqa: BLE001
            print(f"[cascade] mouth stage disabled: {e}")

    tc = cfg.get("temporal", {})
    temporal = TemporalAggregator(
        eye_closed_thresh=tc.get("eye_closed_thresh", 0.5),
        blink_min_frames=tc.get("blink_min_frames", 1),
        perclos_window_sec=tc.get("perclos_window_sec", 60.0),
        yawn_min_duration_sec=tc.get("yawn_min_duration_sec", 1.0),
    )

    return CascadePipeline(detector, head_pose, eye_state, gaze, mouth, temporal)
