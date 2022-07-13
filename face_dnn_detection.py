# smile_detect.py
# all about face detection
# benchmark results (speed & accuracy) depends on parameters passed to functions
# 6.29.22

import cv2
import numpy as np
import time


def show_detection(image, face, confidence):
    # draw rectangles
    # print(faces)
    (x1, y1, x2, y2) = face
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 4)
    # text shadow
    cv2.putText(image, '{:0.2%}'.format(confidence), (x1+1, y1 - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA, bottomLeftOrigin=False)
    # text
    cv2.putText(image, '{:0.2%}'.format(confidence), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 250, 0), 2, cv2.LINE_AA,
                bottomLeftOrigin=False)
    return image


img = cv2.imread('./smiths1.jpg')
# convert grayscale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# blur
img_gray = cv2.medianBlur(img_gray, 3)

confidence_threshold = 0.44

# output image dimensions
w = img.shape[1]
h = img.shape[0]

iterations = 1
start = time.perf_counter()

# load DNN pre-trained model
net = cv2.dnn.readNetFromCaffe('res10_300x300_ssd_deploy.prototxt', caffeModel='res10_300x300_ssd_iter_140000_fp16.caffemodel')

# massage our input data to fit model
blob = cv2.dnn.blobFromImage(img, 1.0, size=(300,300), mean=[104.0, 117.0, 123.0], swapRB=False, crop=False)

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
        show_detection(img, box.astype("int"), confidence)

end = time.perf_counter()
print("Face Detection DNN: {0} msec".format(((end-start) / iterations) * 1000))

# display image
# cv2.imshow('Original', img_gray)
# cv2.waitKey()

cv2.imshow('Face DNN', img)
cv2.waitKey()
