# face.py
# all about face detection
# 6.29.22

import cv2
import numpy as np

def show_detection(image, faces):
    # draw rectangles
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 8)
        cv2.putText(image, 'Face', (x,y-20), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0,255,0), 3, cv2.LINE_AA, bottomLeftOrigin=False)
    return image


img = cv2.imread('./thecure.jpg')
# img = cv2.imread('./depeche.jpg')
img = cv2.imread('./aha.jpg')
# img = cv2.imread('./strawberry.jpg')
# img = cv2.imread('./echo.jpg')

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

# next step, perform detection
# detects faces/objects and returns them as list of rectangles
faces_alt2 = cas_alt2.detectMultiScale(img_gray, minNeighbors=5)
faces_default = cas_default.detectMultiScale(img_gray, scaleFactor=None, minNeighbors=10, minSize=(60,60))

# for getFacesHAAR(), get rid of 1-dimension array by calling np.squeeze()
retval, faces_haar_alt2 = cv2.face.getFacesHAAR(img, "./haarcascade/haarcascade_frontalface_alt2.xml")
retval, faces_haar_default = cv2.face.getFacesHAAR(img, "./haarcascade/haarcascade_frontalface_default.xml")
faces_haar_alt2 = np.squeeze(faces_haar_alt2)
faces_haar_default = np.squeeze(faces_haar_default)

# draw boxes around faces
img_faces_alt2 = show_detection(img.copy(), faces_alt2)
img_faces_default = show_detection(img.copy(), faces_default)
img_faces_haar_alt2 = show_detection(img.copy(), faces_haar_alt2)
img_faces_haar_default = show_detection(img.copy(), faces_haar_default)

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