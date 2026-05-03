
import mediapipe as mp
import cv2
# import gaze
import numpy as np


def gaze(frame, points):
    '''
    2D image points.
    relative takes mediapipe points that normelized to [-1, 1] and returns image points
    at (x,y) format
    '''
    image_points = np.array([
        relative(points.landmark[4], frame.shape),    # Nose tip
        relative(points.landmark[152], frame.shape),  # Chin
        relative(points.landmark[263], frame.shape),  # Left eye left corner
        relative(points.landmark[33], frame.shape),   # Right eye right corner
        relative(points.landmark[287], frame.shape),  # Left Mouth corner
        relative(points.landmark[57], frame.shape)    # Right mouth corner
    ], dtype="double")

    # 3D model points.
    model_points = np.array([
        (0.0, 0.0, 0.0),       # Nose tip
        (0, -63.6, -12.5),     # Chin
        (-43.3, 32.7, -26),    # Left eye left corner
        (43.3, 32.7, -26),     # Right eye right corner
        (-28.9, -28.9, -24.1), # Left Mouth corner
        (28.9, -28.9, -24.1)   # Right mouth corner
    ])
    '''
    3D model eye points
    The center of the eye ball
    '''
    Eye_ball_center_right = np.array([[-29.05],[32.7],[-39.5]])
    Eye_ball_center_left = np.array([[29.05],[32.7],[-39.5]])

mp_face_mesh = mp.solutions.face_mesh # initialize the face mesh model

# camera stream:
cap = cv2.VideoCapture(1)

# video source
cap = cv2.VideoCapture("/home/orinnano/DMS/Driver-attentiveness-monitoring-system/mediapipe_workspace/facial_landmarks/person.webm")   
with mp_face_mesh.FaceMesh(
        max_num_faces=1,                            # number of faces to track in each frame
        refine_landmarks=True,                      # includes iris landmarks in the face mesh model
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:
    while cap.isOpened():
        success, image = cap.read()
        if not success:                            # no frame input
            print("Ignoring empty camera frame.")
            continue
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # frame to RGB for the face-mesh model
        results = face_mesh.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # if results.multi_face_landmarks:
            # gaze.gaze(image, results.multi_face_landmarks[0])

        cv2.imshow('output window', image)
        if cv2.waitKey(2) & 0xFF == 27:          
            break
cap.release()