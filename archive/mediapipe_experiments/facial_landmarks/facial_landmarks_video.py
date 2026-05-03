import mediapipe as mp
import cv2

# def gstreamer_pipeline(
#     capture_width=1920,
#     capture_height=1080,
#     display_width=960,
#     display_height=540,
#     framerate=30,
#     flip_method=0,
# ):
#     return (
#         "nvarguscamerasrc ! "
#         "video/x-raw(memory:NVMM), "
#         "width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
#         "nvvidconv flip-method=%d ! "
#         "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
#         "videoconvert ! "
#         "video/x-raw, format=(string)BGR ! appsink drop=True"
#         % (
#             capture_width,
#             capture_height,
#             framerate,
#             flip_method,
#             display_width,
#             display_height,
#         )
#     )



# Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh =mp_face_mesh.FaceMesh()


# cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)   # for camera
cap = cv2.VideoCapture("/home/orinnano/DMS/Driver-attentiveness-monitoring-system/mediapipe_workspace/facial_landmarks/person.webm")   # for video


while True:
    ret, image = cap.read()

    if ret is not True:
        break
    height, width, _ = image.shape
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Facial landmarks
    result = face_mesh.process(rgb_image)

    for facial_landmarks in result.multi_face_landmarks:
        for i in range(0, 468):
            pt1 = facial_landmarks.landmark[i]
            x = int(pt1.x * width)
            y = int(pt1.y * height)

            cv2.circle(image, (x, y), 2, (100, 100, 0), -1)

    cv2.imshow("Image", image)
    cv2.waitKey(1)