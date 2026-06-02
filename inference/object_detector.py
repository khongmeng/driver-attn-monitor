import os
from dataclasses import dataclass

import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# Objects worth flagging for a driver-monitoring context. A phone in the
# cabin is the headline distraction cue; cup is a mild secondary signal.
DISTRACTION_OBJECTS = {"cell phone"}


@dataclass
class DetectedObject:
    label: str
    score: float
    bbox:  tuple   # (x0, y0, x1, y1) pixels
    is_distraction: bool


class ObjectDetector:
    """MediaPipe Tasks object detector (EfficientDet-Lite, 80 COCO classes).

    Runs alongside the FaceMesh analyzer purely to test the cabin view —
    it does not yet feed the DriverState classifier.
    """

    def __init__(self, cfg: dict):
        model_path = cfg['model_path']
        if not os.path.isabs(model_path):
            root = os.path.join(os.path.dirname(__file__), '..')
            model_path = os.path.normpath(os.path.join(root, model_path))
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Object-detector model not found: {model_path}\n"
                "Download it with:\n"
                "  curl -sL -o models/efficientdet_lite0.tflite \\\n"
                "    https://storage.googleapis.com/mediapipe-models/object_detector/"
                "efficientdet_lite0/int8/1/efficientdet_lite0.tflite")

        options = vision.ObjectDetectorOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            score_threshold=cfg.get('score_threshold', 0.35),
            max_results=cfg.get('max_results', 5),
        )
        self._detector = vision.ObjectDetector.create_from_options(options)

    def process(self, frame_bgr: np.ndarray, timestamp_ms: int) -> list:
        rgb = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(frame_bgr[:, :, ::-1]),
        )
        result = self._detector.detect_for_video(rgb, timestamp_ms)
        out = []
        for det in result.detections:
            cat = det.categories[0]
            bb  = det.bounding_box
            x0, y0 = int(bb.origin_x), int(bb.origin_y)
            x1, y1 = x0 + int(bb.width), y0 + int(bb.height)
            label = cat.category_name or "object"
            out.append(DetectedObject(
                label=label,
                score=float(cat.score),
                bbox=(x0, y0, x1, y1),
                is_distraction=label in DISTRACTION_OBJECTS,
            ))
        return out
