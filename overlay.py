# overlay.py
# how to overlay images (on top of faces)
# 7/13/22

import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('../_models/haarcascades/haarcascade_frontalface_alt.xml')
if face_cascade.empty():
    raise IOError("Unable to load cascade network file")

face_mask = cv2.imread('./anime-mask.jpg', cv2.IMREAD_COLOR)
h_mas, w_mask = face_mask.shape[:2]

cap = cv2.VideoCapture(0)
scaling_factor = 0.5

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # find faces
    face_rects = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in face_rects:
        if h > 0 and w > 0:
            # found a face, adjust height and weight
            # play around with values
            h, w = int(1.1 * h), int(0.75 * w)
            y -= int(0.1 * h)
            x += int(0.15 * w)

            # extract roi from original image
            frame_roi = frame[y:y + h, x:x + w]
            cv2.imshow('frame_roi', frame_roi)
            cv2.waitKey()

            # resize mask to be same size as detected face
            face_mask_small = cv2.resize(face_mask, (w, h), interpolation=cv2.INTER_AREA)
            cv2.imshow('face_mask_small', face_mask_small)
            cv2.waitKey()

            # convert color mask image to grayscale to threshold it
            gray_mask = cv2.cvtColor(face_mask_small, cv2.COLOR_BGR2GRAY)
            ret, mask = cv2.threshold(gray_mask, 180, 255, cv2.THRESH_BINARY_INV)
            cv2.imshow('mask after threshold', mask)
            cv2.waitKey()

            # create an inverse mask
            mask_inv = cv2.bitwise_not(mask)
            cv2.imshow('inverse mask after threshold', mask_inv)
            cv2.waitKey()

            # use the mask to extract the face region of interet
            masked_face = cv2.bitwise_and((face_mask_small), face_mask_small, mask=mask)
            cv2.imshow('AND operation, face_mask_small and MASK', masked_face)
            cv2.waitKey()

            # use the inverse mask to get remaining part of image
            masked_frame = cv2.bitwise_and(frame_roi, frame_roi, mask=mask_inv)
            cv2.imshow('masked frame ', masked_frame)
            cv2.waitKey()

            # add the two images together
            frame[y:y + h, x:x + w] = cv2.add(masked_frame, masked_face)

    cv2.imshow('Face with mask', frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
