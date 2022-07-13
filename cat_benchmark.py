# face.py
# all about face detection
# benchmark results (speed & accuracy) depends on parameters passed to functions
# 6.29.22

import cv2
import numpy as np
import time

def show_detection(image, faces):
    # draw rectangles
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 8)
        cv2.putText(image, 'Cat', (x,y-20), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0,255,0), 2, cv2.LINE_AA, bottomLeftOrigin=False)
    return image


img = cv2.imread('./cats6.jpg')

# convert grayscale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# opencv provides 4 classifiers to use for frontal face detection
# frontalface_alt.xml
# frontalface_alt2.xml
# frontalface_alt_tree.xml
# frontalface_default.xml

# load cascade classifiers, filepath + file

cat_default = cv2.CascadeClassifier("./haarcascade/haarcascade_frontalcatface.xml")
cat_extended = cv2.CascadeClassifier("./haarcascade/haarcascade_frontalcatface_extended.xml")

iterations = 5

# compare speeds of 4 algos

# Best Performing/Fastest Algo for CATS
start = time.perf_counter()
catfaces_default = cat_default.detectMultiScale(img_gray, minNeighbors=1)
# faces_alt2 = cas_alt2.detectMultiScale(img_gray)
for i in range(iterations):
    img_catfaces_default = show_detection(img.copy(), catfaces_default)
end = time.perf_counter()
print("Face detectMultiScale Alt2: {0} msec".format(((end-start) / iterations) * 1000))


start = time.perf_counter()
for i in range(iterations):
    catfaces_extended = cat_extended.detectMultiScale(img_gray, scaleFactor=None, minNeighbors=2, minSize=(215, 215))
    # faces_default = cas_default.detectMultiScale(img_gray)
    img_catfaces_extended = show_detection(img.copy(), catfaces_extended)
end = time.perf_counter()
print("Face detectMultiScale Default: {0} msec".format(((end-start) / iterations) * 1000))


# display image
cv2.imshow('Original', img_gray)
cv2.waitKey()

cv2.imshow('Cat Face Default', img_catfaces_default)
cv2.waitKey()

cv2.imshow('Cat Face Extended', img_catfaces_extended)
cv2.waitKey()

# cv2.imshow('HAAR Alt2', img_faces_haar_alt2)
# cv2.waitKey()
#
# cv2.imshow('HAAR Default', img_faces_haar_default)
# cv2.waitKey()