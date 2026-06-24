import cv2
import numpy as np
import mediapipe as mp
from dataclasses import dataclass
from typing import Optional

# 6-point 3D face model (mm) used for solvePnP head-pose estimation
_MODEL_POINTS = np.array([
    (0.0,    0.0,    0.0),      # nose tip       [1]
    (0.0,   -330.0, -65.0),     # chin           [199]
    (-225.0, 170.0, -135.0),    # left eye outer [33]
    (225.0,  170.0, -135.0),    # right eye outer[263]
    (-150.0,-150.0, -125.0),    # left mouth     [61]
    (150.0, -150.0, -125.0),    # right mouth    [291]
], dtype=np.float64)

_POSE_IDS     = [1, 199, 33, 263, 61, 291]
_LEFT_EYE_IDS = [33, 160, 158, 133, 153, 144]
_RIGHT_EYE_IDS= [362, 385, 387, 263, 373, 380]


@dataclass
class FaceFeatures:
    bbox:        tuple           # (x0, y0, x1, y1) pixels
    ear:         float           # avg eye aspect ratio
    yaw:         float           # head yaw  degrees
    pitch:       float           # head pitch degrees
    roll:        float           # head roll  degrees
    gaze_offset: float           # normalized iris offset from eye midpoint
    nose_tip:    tuple           # (x, y) for head-pose arrow base
    nose_dir:    tuple           # (x, y) for head-pose arrow tip
    left_iris:   np.ndarray      # center (x, y)
    right_iris:  np.ndarray      # center (x, y)
    gaze_end:    tuple           # (x, y) end of gaze arrow


class FaceAnalyzer:
    def __init__(self, cfg: dict, gaze_cfg: dict = None):
        self._mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=cfg.get('max_faces', 1),
            refine_landmarks=cfg.get('refine_landmarks', True),
        )
        gaze_cfg          = gaze_cfg or {}
        self._gaze_gain   = gaze_cfg.get('arrow_gain', 220.0)
        self._gaze_dead   = gaze_cfg.get('deadzone', 0.04)
        self._gaze_alpha  = gaze_cfg.get('smoothing', 0.35)
        self._gaze_ema    = None   # EMA of normalized gaze vector (single-face cabin cam)

    def process(self, frame_bgr: np.ndarray) -> list:
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self._mesh.process(rgb)
        if not results.multi_face_landmarks:
            return []
        return [f for lm in results.multi_face_landmarks
                if (f := self._extract(lm.landmark, w, h)) is not None]

    def _extract(self, lm, w, h) -> Optional[FaceFeatures]:
        def px(i): return lm[i].x * w, lm[i].y * h

        # Bounding box
        xs = [l.x * w for l in lm]
        ys = [l.y * h for l in lm]
        bbox = (max(0, int(min(xs))), max(0, int(min(ys))),
                min(w - 1, int(max(xs))), min(h - 1, int(max(ys))))

        # EAR
        def eye_pts(ids): return np.array([px(i) for i in ids], dtype=np.float32)
        ear = (_ear(eye_pts(_LEFT_EYE_IDS)) + _ear(eye_pts(_RIGHT_EYE_IDS))) / 2.0

        # Head pose
        img_pts = np.array([px(i) for i in _POSE_IDS], dtype=np.float64)
        cam_mat = np.array([[w, 0, w/2], [0, w, h/2], [0, 0, 1]], dtype=np.float64)
        ok, rvec, tvec = cv2.solvePnP(_MODEL_POINTS, img_pts, cam_mat,
                                       np.zeros((4, 1)), flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            return None

        rot, _ = cv2.Rodrigues(rvec)
        sy    = np.sqrt(rot[0, 0] ** 2 + rot[1, 0] ** 2)
        pitch = np.degrees(np.arctan2(-rot[2, 0], sy))
        yaw   = np.degrees(np.arctan2(rot[2, 1],  rot[2, 2]))
        roll  = np.degrees(np.arctan2(rot[1, 0],  rot[0, 0]))

        nose_tip = (int(img_pts[0][0]), int(img_pts[0][1]))
        end2d, _ = cv2.projectPoints(np.array([[0.0, 0.0, 150.0]]),
                                      rvec, tvec, cam_mat, np.zeros((4, 1)))
        nose_dir = (int(end2d[0][0][0]), int(end2d[0][0][1]))

        # Iris / gaze
        left_iris  = np.mean([px(i) for i in range(468, 473)], axis=0)
        right_iris = np.mean([px(i) for i in range(473, 478)], axis=0)

        # Per-eye iris offset from the eye-corner midpoint, normalized by eye
        # width so it is resolution- and head-distance-independent. Averaging
        # both eyes halves the per-frame landmark noise.  (#1 proportional, #4 both eyes)
        def eye_offset(iris, outer_id, inner_id):
            outer = np.array(px(outer_id))
            inner = np.array(px(inner_id))
            width = np.linalg.norm(outer - inner) + 1e-6
            return (iris - (outer + inner) / 2.0) / width

        gv = (eye_offset(left_iris, 33, 133) +
              eye_offset(right_iris, 263, 362)) / 2.0

        # Temporal smoothing: EMA of the normalized gaze vector.  (#3)
        if self._gaze_ema is None:
            self._gaze_ema = gv
        else:
            self._gaze_ema = (self._gaze_alpha * gv +
                              (1.0 - self._gaze_alpha) * self._gaze_ema)
        gaze_vec    = self._gaze_ema
        gaze_offset = float(np.linalg.norm(gaze_vec))

        # Deadzone: below threshold the iris is centered (looking straight) and
        # the direction is dominated by noise, so draw no arrow.  (#2)
        if gaze_offset < self._gaze_dead:
            gaze_end = (int(left_iris[0]), int(left_iris[1]))
        else:
            gaze_end = (int(left_iris[0] + gaze_vec[0] * self._gaze_gain),
                        int(left_iris[1] + gaze_vec[1] * self._gaze_gain))

        return FaceFeatures(
            bbox=bbox, ear=ear,
            yaw=yaw, pitch=pitch, roll=roll,
            gaze_offset=gaze_offset,
            nose_tip=nose_tip, nose_dir=nose_dir,
            left_iris=left_iris, right_iris=right_iris,
            gaze_end=gaze_end,
        )


def _ear(pts: np.ndarray) -> float:
    A = np.linalg.norm(pts[1] - pts[5])
    B = np.linalg.norm(pts[2] - pts[4])
    C = np.linalg.norm(pts[0] - pts[3]) + 1e-6
    return (A + B) / (2.0 * C)
