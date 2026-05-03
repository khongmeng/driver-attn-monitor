import cv2
import numpy as np
from .state_detector import DriverState, StateResult
from .face_analyzer import FaceFeatures

_STATE_COLOR = {
    DriverState.FOCUSED:    (0, 200, 0),
    DriverState.DROWSY:     (0, 60, 255),
    DriverState.DISTRACTED: (0, 165, 255),
    DriverState.NO_FACE:    (120, 120, 120),
}


def draw(frame: np.ndarray, features: list, result: StateResult, fps: float) -> np.ndarray:
    color = _STATE_COLOR[result.state]
    w     = frame.shape[1]

    for f in features:
        x0, y0, x1, y1 = f.bbox
        cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
        cv2.arrowedLine(frame, f.nose_tip, f.nose_dir, color, 2, tipLength=0.3)
        cv2.circle(frame, tuple(f.left_iris.astype(int)),  3, (0, 0, 255), -1)
        cv2.circle(frame, tuple(f.right_iris.astype(int)), 3, (0, 0, 255), -1)
        cv2.arrowedLine(frame, tuple(f.left_iris.astype(int)), f.gaze_end,
                        (0, 0, 255), 2, tipLength=0.3)

    _hud(frame, f"State:   {result.state.value}",            (10,  30), color, 0.8)
    _hud(frame, f"EAR:     {result.ear:.3f}",                (10,  58), (200, 200, 200))
    _hud(frame, f"PERCLOS: {result.perclos:.1%}",            (10,  86), (200, 200, 200))
    _hud(frame, f"Yaw:     {result.yaw:.1f}°",               (10, 114), (200, 200, 200))
    _hud(frame, f"Pitch:   {result.pitch:.1f}°",             (10, 142), (200, 200, 200))
    _hud(frame, f"Blinks:  {result.blink_count}  {result.blink_rate:.1f}/min", (10, 170), (200, 200, 200))
    _hud(frame, f"FPS: {fps:.1f}", (w - 120, 30), (180, 180, 180), 0.6)
    return frame


def _hud(frame, text, pos, color, scale=0.65):
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3)
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1)
