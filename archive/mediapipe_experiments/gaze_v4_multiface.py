import cv2
import numpy as np
import mediapipe as mp

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

cap = cv2.VideoCapture("/home/orinnano/DMS/Driver-attentiveness-monitoring-system/mediapipe_workspace/video/TWO_face.webm")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    if results.multi_face_landmarks:
        for face_idx, face in enumerate(results.multi_face_landmarks):
            color = COLORS[face_idx % len(COLORS)]

            # ====== Bounding box for the face ======
            xs = [lm.x * w for lm in face.landmark]
            ys = [lm.y * h for lm in face.landmark]
            x_min, x_max = int(min(xs)), int(max(xs))
            y_min, y_max = int(min(ys)), int(max(ys))

            # Draw face rectangle
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)

            # Label text above the box
            label = f"Face {face_idx}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            text_x = x_min
            text_y = max(y_min - 10, th + 5)
            cv2.rectangle(frame, (text_x - 1, text_y - th - 4), (text_x + tw + 1, text_y + 2), color, -1)
            cv2.putText(frame, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

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
            nose_end_2d, _ = cv2.projectPoints(nose_end_3d, rvec, tvec, camera_matrix, dist_coeffs)
            nose_end_2d = (int(nose_end_2d[0][0][0]), int(nose_end_2d[0][0][1]))
            cv2.arrowedLine(frame, nose_tip, nose_end_2d, color, 2, tipLength=0.3)

            # ====== Iris / eye direction ======
            left_iris = np.array([(face.landmark[i].x * w, face.landmark[i].y * h) for i in range(468, 473)])
            right_iris = np.array([(face.landmark[i].x * w, face.landmark[i].y * h) for i in range(473, 478)])
            left_center = np.mean(left_iris, axis=0)
            right_center = np.mean(right_iris, axis=0)

            cv2.circle(frame, tuple(left_center.astype(int)), 3, (0, 0, 255), -1)
            cv2.circle(frame, tuple(right_center.astype(int)), 3, (0, 0, 255), -1)

            left_outer = np.array([face.landmark[33].x * w, face.landmark[33].y * h])
            left_inner = np.array([face.landmark[133].x * w, face.landmark[133].y * h])
            left_mid = (left_outer + left_inner) / 2.0
            gaze_vec = left_center - left_mid
            norm = np.linalg.norm(gaze_vec)
            if norm < 1e-6:continue
            gaze_vec = (gaze_vec / norm) * 80
            gaze_end = (int(left_center[0] + gaze_vec[0]), int(left_center[1] + gaze_vec[1]))
            cv2.arrowedLine(frame, tuple(left_center.astype(int)), gaze_end, (0, 0, 255), 2, tipLength=0.3)

    cv2.imshow("Multi-face Gaze + Bounding Boxes", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()