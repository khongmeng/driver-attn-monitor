import cv2
import numpy as np
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# Generic 3D facial model points (based on average human head)
MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),            # Nose tip
    (0.0, -330.0, -65.0),       # Chin
    (-225.0, 170.0, -135.0),    # Left eye left corner
    (225.0, 170.0, -135.0),     # Right eye right corner
    (-150.0, -150.0, -125.0),   # Left mouth corner
    (150.0, -150.0, -125.0)     # Right mouth corner
], dtype=np.float64)

LANDMARK_IDS = [1, 199, 33, 263, 61, 291]  # Landmarks corresponding to above points

cap = cv2.VideoCapture("/home/orinnano/DMS/Driver-attentiveness-monitoring-system/mediapipe_workspace/video/TWO_face.webm")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    h, w = frame.shape[:2]

    if results.multi_face_landmarks:
        face = results.multi_face_landmarks[0]
        image_points = np.array([(face.landmark[i].x * w, face.landmark[i].y * h) for i in LANDMARK_IDS], dtype=np.float64)

        # Camera matrix
        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1))

        # Estimate head pose
        success, rotation_vec, translation_vec = cv2.solvePnP(
            MODEL_POINTS, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
        )

        # Project a 3D point (nose tip forward)
        nose_end_3d = np.array([[0, 0, 150.0]])
        nose_end_2d, _ = cv2.projectPoints(nose_end_3d, rotation_vec, translation_vec, camera_matrix, dist_coeffs)
        p1 = (int(image_points[0][0]), int(image_points[0][1]))  # nose tip
        p2 = (int(nose_end_2d[0][0][0]), int(nose_end_2d[0][0][1]))

        # Draw head direction
        cv2.arrowedLine(frame, p1, p2, (0, 255, 0), 3, tipLength=0.3)
        # cv2.putText(frame, "Head direction", (p2[0]+10, p2[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        # Iris tracking
        left_iris_indices = range(468, 473)
        right_iris_indices = range(473, 478)

        left_iris = np.array([(face.landmark[i].x * w, face.landmark[i].y * h) for i in left_iris_indices])
        right_iris = np.array([(face.landmark[i].x * w, face.landmark[i].y * h) for i in right_iris_indices])
        left_center = np.mean(left_iris, axis=0).astype(int)
        right_center = np.mean(right_iris, axis=0).astype(int)

        cv2.circle(frame, tuple(left_center), 3, (0, 0, 255), -1)
        cv2.circle(frame, tuple(right_center), 3, (0, 0, 255), -1)

        # Simple gaze direction estimate — offset from iris to eye corner
        # 33 (left eye outer corner), 133 (left eye inner corner)
        left_eye_outer = np.array([face.landmark[33].x * w, face.landmark[33].y * h])
        left_eye_inner = np.array([face.landmark[133].x * w, face.landmark[133].y * h])

        gaze_dir = left_center - (left_eye_outer + left_eye_inner) / 2
        gaze_dir_norm = gaze_dir / np.linalg.norm(gaze_dir) * 80  # scale "arrow length"

        gaze_end = (int(left_center[0] + gaze_dir_norm[0]), int(left_center[1] + gaze_dir_norm[1]))
        cv2.arrowedLine(frame, tuple(left_center), gaze_end, (0, 0, 255), 3, tipLength=0.3)
        # cv2.putText(frame, "Eye direction", (gaze_end[0]+10, gaze_end[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

    cv2.imshow("Eye + Head Gaze Estimation", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()