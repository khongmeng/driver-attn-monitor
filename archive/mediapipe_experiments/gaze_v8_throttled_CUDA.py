import cv2
import numpy as np
import mediapipe as mp
import time
import os

# ---------- Setup ----------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=5,            # detect multiple faces
    refine_landmarks=True
)

MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),            # Nose tip
    (0.0, -330.0, -65.0),       # Chin
    (-225.0, 170.0, -135.0),    # Left eye left corner
    (225.0, 170.0, -135.0),     # Right eye right corner
    (-150.0, -150.0, -125.0),   # Left mouth corner
    (150.0, -150.0, -125.0)
], dtype=np.float64)

LANDMARK_IDS = [1, 199, 33, 263, 61, 291]

COLORS = [
    (0, 255, 0),
    (0, 165, 255),
    (255, 0, 0),
    (255, 0, 255),
    (0, 255, 255)
]

# Eye landmark indices for EAR (FaceMesh)
LEFT_EYE_IDXS  = [33, 160, 158, 133, 153, 144]     # p1, p2, p3, p4, p5, p6
RIGHT_EYE_IDXS = [362, 385, 387, 263, 373, 380]

# Blink detection params
EAR_THRESH = 0.21          # tune this
EAR_CONSEC_FRAMES = 3      # how many consecutive frames eye must be "closed" to count blink
MAX_FACES = 5

# Per-face blink state
blink_counts = [0] * MAX_FACES
closed_frames = [0] * MAX_FACES
start_time = time.time()

# ---------- Age & Gender Estimation (OpenCV DNN) ----------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Age model
AGE_PROTO = os.path.join(SCRIPT_DIR, "age_deploy.prototxt")
AGE_MODEL = os.path.join(SCRIPT_DIR, "age_net.caffemodel")

AGE_BUCKETS = [
    "(0-2)",
    "(4-6)",
    "(8-12)",
    "(15-20)",
    "(25-32)",
    "(38-43)",
    "(48-53)",
    "(60-100)"
]

# Gender model
GENDER_PROTO = os.path.join(SCRIPT_DIR, "gender_deploy.prototxt")
GENDER_MODEL = os.path.join(SCRIPT_DIR, "gender_net.caffemodel")

GENDER_LIST = ["Male", "Female"]

# Mean values used in the original Caffe models (BGR order)
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

age_net = cv2.dnn.readNet(AGE_MODEL, AGE_PROTO)
gender_net = cv2.dnn.readNet(GENDER_MODEL, GENDER_PROTO)

# ---- Try to use CUDA on Jetson (falls back silently if not available) ----
try:
    age_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    age_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
    gender_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    gender_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
    print("Using CUDA for age/gender DNNs")
except cv2.error as e:
    print("CUDA not available for OpenCV DNN, using CPU:", e)

# Run age/gender only every N frames
AGE_GENDER_INTERVAL = 15
frame_idx = 0

# Cache last predictions per face (so UI is stable & fast)
last_age_label   = [""] * MAX_FACES
last_gender_label = [""] * MAX_FACES


def gstreamer_pipeline(
        sensor_id=0,
        capture_width=1280,
        capture_height=720,
        display_width=1280,
        display_height=720,
        framerate=30,
        flip_method=0
):
    # If you want extra speed, try 960x540 or 640x480 here
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width=(int){capture_width}, height=(int){capture_height}, "
        f"format=(string)NV12, framerate=(fraction){framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width=(int){display_width}, height=(int){display_height}, format=(string)BGRx ! "
        f"videoconvert ! "
        f"video/x-raw, format=(string)BGR ! appsink"
    )


def eye_aspect_ratio(eye_pts: np.ndarray) -> float:
    """
    eye_pts: shape (6, 2) -> [p1, p2, p3, p4, p5, p6]
    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    """
    if eye_pts.shape != (6, 2):
        return 0.0

    A = np.linalg.norm(eye_pts[1] - eye_pts[5])
    B = np.linalg.norm(eye_pts[2] - eye_pts[4])
    C = np.linalg.norm(eye_pts[0] - eye_pts[3])

    if C < 1e-6:
        return 0.0

    return (A + B) / (2.0 * C)


cap = cv2.VideoCapture(gstreamer_pipeline(sensor_id=0), cv2.CAP_GSTREAMER)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1

    h, w = frame.shape[:2]
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    if results.multi_face_landmarks:
        for face_idx, face in enumerate(results.multi_face_landmarks):
            if face_idx >= MAX_FACES:
                break

            color = COLORS[face_idx % len(COLORS)]

            # ====== Bounding box for the face ======
            xs = [lm.x * w for lm in face.landmark]
            ys = [lm.y * h for lm in face.landmark]
            x_min, x_max = int(min(xs)), int(max(xs))
            y_min, y_max = int(min(ys)), int(max(ys))

            # Clip to frame
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(w - 1, x_max)
            y_max = min(h - 1, y_max)

            # Draw face rectangle
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)

            # ====== Age & Gender estimation on face ROI (throttled) ======
            age_label = last_age_label[face_idx]
            gender_label = last_gender_label[face_idx]

            run_age_gender = (frame_idx % AGE_GENDER_INTERVAL == 0)

            if run_age_gender and x_max > x_min and y_max > y_min:
                face_roi = frame[y_min:y_max, x_min:x_max]
                if face_roi.size > 0:
                    blob = cv2.dnn.blobFromImage(
                        face_roi,
                        1.0,
                        (227, 227),
                        MODEL_MEAN_VALUES,
                        swapRB=False
                    )

                    # Age
                    age_net.setInput(blob)
                    age_preds = age_net.forward()
                    age_idx = int(age_preds[0].argmax())
                    age_label = AGE_BUCKETS[age_idx]
                    last_age_label[face_idx] = age_label  # cache

                    # Gender
                    gender_net.setInput(blob)
                    gender_preds = gender_net.forward()
                    gender_idx = int(gender_preds[0].argmax())
                    gender_label = GENDER_LIST[gender_idx]
                    last_gender_label[face_idx] = gender_label  # cache

            # ====== Label text above the box (Face + Gender + Age) ======
            if gender_label and age_label:
                label = f"Face {face_idx} {gender_label} {age_label}"
            elif age_label:
                label = f"Face {face_idx} {age_label}"
            else:
                label = f"Face {face_idx}"

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            text_x = x_min
            text_y = max(y_min - 10, th + 5)

            cv2.rectangle(
                frame,
                (text_x - 1, text_y - th - 4),
                (text_x + tw + 1, text_y + 2),
                color,
                -1
            )
            cv2.putText(
                frame,
                label,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )

            # ====== Head pose estimation ======
            image_points = np.array(
                [(face.landmark[i].x * w, face.landmark[i].y * h) for i in LANDMARK_IDS],
                dtype=np.float64
            )
            focal_length = w
            center = (w / 2, h / 2)
            camera_matrix = np.array([[focal_length, 0, center[0]],
                                      [0, focal_length, center[1]],
                                      [0, 0, 1]], dtype=np.float64)
            dist_coeffs = np.zeros((4, 1))

            ok, rvec, tvec = cv2.solvePnP(
                MODEL_POINTS, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
            )
            if not ok:
                continue

            nose_tip = (int(image_points[0][0]), int(image_points[0][1]))
            nose_end_3d = np.array([[0, 0, 150.0]])
            nose_end_2d, _ = cv2.projectPoints(
                nose_end_3d, rvec, tvec, camera_matrix, dist_coeffs
            )
            nose_end_2d = (int(nose_end_2d[0][0][0]), int(nose_end_2d[0][0][1]))
            cv2.arrowedLine(frame, nose_tip, nose_end_2d, color, 2, tipLength=0.3)

            # ====== Iris / eye direction ======
            left_iris = np.array(
                [(face.landmark[i].x * w, face.landmark[i].y * h) for i in range(468, 473)]
            )
            right_iris = np.array(
                [(face.landmark[i].x * w, face.landmark[i].y * h) for i in range(473, 478)]
            )

            left_center = np.mean(left_iris, axis=0)
            right_center = np.mean(right_iris, axis=0)

            cv2.circle(frame, tuple(left_center.astype(int)), 3, (0, 0, 255), -1)
            cv2.circle(frame, tuple(right_center.astype(int)), 3, (0, 0, 255), -1)

            left_outer = np.array(
                [face.landmark[33].x * w, face.landmark[33].y * h]
            )
            left_inner = np.array(
                [face.landmark[133].x * w, face.landmark[133].y * h]
            )
            left_mid = (left_outer + left_inner) / 2.0
            gaze_vec = left_center - left_mid
            norm = np.linalg.norm(gaze_vec)
            if norm < 1e-6:
                continue
            gaze_vec = (gaze_vec / norm) * 80
            gaze_end = (
                int(left_center[0] + gaze_vec[0]),
                int(left_center[1] + gaze_vec[1])
            )
            cv2.arrowedLine(
                frame,
                tuple(left_center.astype(int)),
                gaze_end,
                (0, 0, 255),
                2,
                tipLength=0.3
            )

            # ====== Blink detection (EAR) ======
            left_eye_pts = np.array(
                [[face.landmark[i].x * w, face.landmark[i].y * h] for i in LEFT_EYE_IDXS],
                dtype=np.float32
            )
            right_eye_pts = np.array(
                [[face.landmark[i].x * w, face.landmark[i].y * h] for i in RIGHT_EYE_IDXS],
                dtype=np.float32
            )

            ear_left = eye_aspect_ratio(left_eye_pts)
            ear_right = eye_aspect_ratio(right_eye_pts)
            ear = (ear_left + ear_right) / 2.0

            # Update blink state for this face
            if ear < EAR_THRESH:
                closed_frames[face_idx] += 1
            else:
                if closed_frames[face_idx] >= EAR_CONSEC_FRAMES:
                    blink_counts[face_idx] += 1
                closed_frames[face_idx] = 0

            # Compute blink rate (blinks per minute)
            elapsed_minutes = (time.time() - start_time) / 60.0
            if elapsed_minutes > 0:
                blink_rate = blink_counts[face_idx] / elapsed_minutes
            else:
                blink_rate = 0.0

            # Draw blink info just below the face label
            blink_text = f"Blink: {blink_counts[face_idx]}  BR: {blink_rate:.1f}/min"
            cv2.putText(
                frame,
                blink_text,
                (x_min, text_y + th + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )

    cv2.imshow("Multi-face Gaze + Blink + Age/Gender", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()