# face_detection_mediapipe.py
# face detection using mediapipe (blazeface) algorithm
# MediaPipe is fast, but it seems not as accurate as DNN Res10 model algorithm.
# also, tends to miss faces, and the minimum confidence factor does not seem to work
# Res10 algorithm is just a tad bit slower, but more accurate.
# 7/2/22

import time

import cv2
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# For static images:
IMAGE_FILES = ['./smiths1.jpg']

# with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.51) as face_detection:

face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.21)
for idx, file in enumerate(IMAGE_FILES):
    image = cv2.imread(file)

    start_time = time.time()

    # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
    results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Draw face detections of each face.
    if not results.detections:
        continue
    annotated_image = image.copy()
    for detection in results.detections:
        # print('Nose tip:')
        # print(mp_face_detection.get_key_point(detection, mp_face_detection.FaceKeyPoint.NOSE_TIP))
        # print('Right eye:')
        # print(mp_face_detection.get_key_point(detection, mp_face_detection.FaceKeyPoint.RIGHT_EYE))
        drawing_spec = mp_drawing.DrawingSpec(thickness=5, circle_radius=0, color=(0,255,0))
        kp_drawing_spec = mp_drawing.DrawingSpec(color=(0,255,255), circle_radius=0)
        mp_drawing.draw_detection(annotated_image, detection, keypoint_drawing_spec=kp_drawing_spec, bbox_drawing_spec=drawing_spec)
        # print(detection)

    # cv2.imwrite('/tmp/annotated_image' + str(idx) + '.png', annotated_image)
    end_time = time.time()

cv2.imshow('MediaPipe output', annotated_image)
cv2.waitKey(0)
print("Execution time:{:0.4f} seconds".format(end_time - start_time))

    # # For webcam input:
    # cap = cv2.VideoCapture(0)
    # with mp_face_detection.FaceDetection(
    #         model_selection=0, min_detection_confidence=0.5) as face_detection:
    #     while cap.isOpened():
    #         success, image = cap.read()
    #         if not success:
    #             print("Ignoring empty camera frame.")
    #             # If loading a video, use 'break' instead of 'continue'.
    #             continue
    #
    #         # To improve performance, optionally mark the image as not writeable to
    #         # pass by reference.
    #         image.flags.writeable = False
    #         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #         results = face_detection.process(image)
    #
    #         # Draw the face detection annotations on the image.
    #         image.flags.writeable = True
    #         image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    #         if results.detections:
    #             for detection in results.detections:
    #                 mp_drawing.draw_detection(image, detection)
    #         # Flip the image horizontally for a selfie-view display.
    #         cv2.imshow('MediaPipe Face Detection', cv2.flip(image, 1))
    #         if cv2.waitKey(5) & 0xFF == 27:
    #             break
    # cap.release()
