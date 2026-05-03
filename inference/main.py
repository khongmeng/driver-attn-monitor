import cv2
import time
import yaml
import os

from .camera import Camera
from .face_analyzer import FaceAnalyzer
from .state_detector import StateDetector
from .overlay import draw


def load_config() -> dict:
    cfg_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config()

    cam      = Camera(cfg['camera'])
    analyzer = FaceAnalyzer(cfg['face_mesh'])
    detector = StateDetector(cfg)

    fps_timer   = time.time()
    fps         = 0.0
    frame_count = 0

    try:
        while True:
            ret, frame = cam.read()
            if not ret:
                break

            features = analyzer.process(frame)
            result   = detector.update(features)
            frame    = draw(frame, features, result, fps)

            cv2.imshow(cfg['display']['window_title'], frame)
            if cv2.waitKey(1) & 0xFF == 27:   # Esc to quit
                break

            frame_count += 1
            now = time.time()
            if now - fps_timer >= 1.0:
                fps         = frame_count / (now - fps_timer)
                frame_count = 0
                fps_timer   = now
    finally:
        cam.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
