# smile_detect.py
# all about face detection
# benchmark results (speed & accuracy) depends on parameters passed to functions
# 6.29.22

import cv2
import numpy as np
import time

def show_detection(image, faces):
    # draw rectangles
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 4)
        # text shadow
        cv2.putText(image, 'Smile', (x+2, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA, bottomLeftOrigin=False)
        # text
        cv2.putText(image, 'Smile', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (100, 250, 100), 2, cv2.LINE_AA,
                    bottomLeftOrigin=False)
    return image


img = cv2.imread('./smile-group.jpg')
# convert grayscale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# blur
img_gray = cv2.medianBlur(img_gray, 3)

# load cascade classifiers, filepath + file
smile_default = cv2.CascadeClassifier("./haarcascade/haarcascade_smile.xml")

iterations = 1
start = time.perf_counter()
for i in range(iterations):
    smilingfaces_default = smile_default.detectMultiScale(img_gray, minNeighbors=85, minSize=(50, 10), maxSize=(100, 100))
    img_smilingfaces_default = show_detection(img.copy(), smilingfaces_default)
end = time.perf_counter()
print("Smiling Faces Default: {0} msec".format(((end-start) / iterations) * 1000))

# display image
cv2.imshow('Original', img_gray)
cv2.waitKey()

cv2.imshow('Cat Face Default', img_smilingfaces_default)
cv2.waitKey()
