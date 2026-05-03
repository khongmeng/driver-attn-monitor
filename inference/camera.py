import cv2


def _gstreamer_pipeline(sensor_id, width, height, framerate, flip_method):
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width=(int){width}, height=(int){height}, "
        f"format=(string)NV12, framerate=(fraction){framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width=(int){width}, height=(int){height}, format=(string)BGRx ! "
        f"videoconvert ! "
        f"video/x-raw, format=(string)BGR ! appsink"
    )


class Camera:
    def __init__(self, cfg: dict):
        pipeline = _gstreamer_pipeline(
            cfg['sensor_id'], cfg['width'], cfg['height'],
            cfg['framerate'], cfg['flip_method'],
        )
        self._cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if not self._cap.isOpened():
            raise RuntimeError("Camera failed to open. Check sensor_id and GStreamer pipeline.")

    def read(self):
        return self._cap.read()

    def release(self):
        self._cap.release()
