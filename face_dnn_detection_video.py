# smile_detect.py
# all about face detection
# benchmark results (speed & accuracy) depends on parameters passed to functions
# 6.29.22

import cv2
import numpy as np
import time

USE_CUDA = False

iterations = 1
start = time.perf_counter()

def show_detection(image, face, confidence):
    # draw rectangles
    # print(faces)
    (x1, y1, x2, y2) = face
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 4)
    # text shadow
    cv2.putText(image, '{:0.2%}'.format(confidence), (x1+1, y1 - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA, bottomLeftOrigin=False)
    # text
    cv2.putText(image, '{:0.2%}'.format(confidence), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 250, 0), 2, cv2.LINE_AA,
                bottomLeftOrigin=False)
    return image


cap = cv2.VideoCapture('./video/fascination.mp4')  # video file
# cap = cv2.VideoCapture(0)   # usb camera
confidence_threshold = 0.35

# output image dimensions
w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

# load DNN pre-trained model
# net = cv2.dnn.readNetFromCaffe('res10_300x300_ssd_deploy.prototxt', caffeModel='res10_300x300_ssd_iter_140000_fp16.caffemodel')
net = cv2.dnn.readNetFromCaffe('./ssd/res10_300x300_ssd_deploy.prototxt',
                               caffeModel='./ssd/res10_300x300_ssd_iter_140000.caffemodel')
# use cuda
if USE_CUDA:
    _ = net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    _ = net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

while cv2.waitKey(1) != ord(' ') and cap.isOpened():  # space bar
    # iterations = 1
    # start = time.perf_counter()

    ret, frame = cap.read()
    if not ret:
        break

    # massage our input data to fit model
    blob = cv2.dnn.blobFromImage(frame, 1.0, size=(300,300), mean=[104.0, 117.0, 123.0], swapRB=False, crop=False)

    # feed our blob as input to our net model
    net.setInput(blob)

    # inference, forward
    detections = net.forward()

    # iterate over all found faces in detections
    detected_faces_count = 0
    for i in range(detections.shape[2]):
        # get confidence value
        confidence = detections[0,0,i,2]

        # only consider this item if confidence > acceptable criteria
        if confidence > confidence_threshold:
            # count number of faces detected
            detected_faces_count = detected_faces_count + 1
            # get box coordinates for this face
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            # draw rectangle and confidence value
            show_detection(frame, box.astype("int"), confidence)

    # end = time.perf_counter()
    # print("Face Detection DNN: {0} msec".format(((end - start) / iterations) * 1000))
    cv2.imshow('Face DNN', frame)

end = time.perf_counter()
print("Face Detection DNN: {0} msec".format(((end-start) / iterations) * 1000))
cv2.destroyAllWindows()
