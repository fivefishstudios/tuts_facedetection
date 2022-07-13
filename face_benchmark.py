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
        cv2.putText(image, 'Face', (x,y-20), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0,255,0), 2, cv2.LINE_AA, bottomLeftOrigin=False)
    return image


# img = cv2.imread('./thecure.jpg')
# img = cv2.imread('./depeche.jpg')
# img = cv2.imread('./aha.jpg')
# img = cv2.imread('./strawberry.jpg')
# img = cv2.imread('./echo.jpg')
# img = cv2.imread('./ultravox.jpg')
# img = cv2.imread('./seagulls.jpg')
# img = cv2.imread('./talktalk.jpg')
# img = cv2.imread('./flock2.jpg')
# img = cv2.imread('./more2lose.jpg')
# img = cv2.imread('./spandau.jpg')
img = cv2.imread('./soccer1.jpg')

# convert grayscale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# opencv provides 4 classifiers to use for frontal face detection
# frontalface_alt.xml
# frontalface_alt2.xml
# frontalface_alt_tree.xml
# frontalface_default.xml

# load cascade classifiers, filepath + file

cas_alt2 = cv2.CascadeClassifier("./haarcascade/haarcascade_frontalface_alt2.xml")
cas_default = cv2.CascadeClassifier("./haarcascade/haarcascade_frontalface_default.xml")

iterations = 5

# compare speeds of 4 algos
start = time.perf_counter()
faces_alt2 = cas_alt2.detectMultiScale(img_gray, minNeighbors=5)
# faces_alt2 = cas_alt2.detectMultiScale(img_gray)
for i in range(iterations):
    img_faces_alt2 = show_detection(img.copy(), faces_alt2)
end = time.perf_counter()
print("Face detectMultiScale Alt2: {0} msec".format(((end-start) / iterations) * 1000))


# Best Perfrming/Fastest Algo
start = time.perf_counter()
for i in range(iterations):
    faces_default = cas_default.detectMultiScale(img_gray, scaleFactor=None, minNeighbors=10, minSize=(60,60))
    # faces_default = cas_default.detectMultiScale(img_gray)
    img_faces_default = show_detection(img.copy(), faces_default)
end = time.perf_counter()
print("Face detectMultiScale Default: {0} msec".format(((end-start) / iterations) * 1000))


start = time.perf_counter()
for i in range(iterations):
    retval, faces_haar_alt2 = cv2.face.getFacesHAAR(img, "./haarcascade/haarcascade_frontalface_alt2.xml")
    faces_haar_alt2 = np.squeeze(faces_haar_alt2)
    img_faces_haar_alt2 = show_detection(img.copy(), faces_haar_alt2)
end = time.perf_counter()
print("Face HAAR Alt2: {0} msec".format(((end-start) / iterations) * 1000))


start = time.perf_counter()
for i in range(iterations):
    retval, faces_haar_default = cv2.face.getFacesHAAR(img, "./haarcascade/haarcascade_frontalface_default.xml")
    faces_haar_default = np.squeeze(faces_haar_default)
    img_faces_haar_default = show_detection(img.copy(), faces_haar_default)
end = time.perf_counter()
print("Face HAAR Default: {0} msec".format(((end-start) / iterations) * 1000))

# display image
cv2.imshow('Original', img_gray)
cv2.waitKey()

cv2.imshow('Face Alt 2', img_faces_alt2)
cv2.waitKey()

cv2.imshow('Face Default', img_faces_default)
cv2.waitKey()

cv2.imshow('HAAR Alt2', img_faces_haar_alt2)
cv2.waitKey()

cv2.imshow('HAAR Default', img_faces_haar_default)
cv2.waitKey()