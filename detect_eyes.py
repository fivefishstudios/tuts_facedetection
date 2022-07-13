# detect_eyes.py
# detect eyes
# 7/13/22

import cv2

face_cascade = cv2.CascadeClassifier('./haarcascade/haarcascade_frontalface_alt.xml')
eye_cascade = cv2.CascadeClassifier('./haarcascade/haarcascade_lefteye_2splits.xml')


if face_cascade.empty():
    raise IOError('Face Cascade not found')

if eye_cascade.empty():
    raise IOError('Eye Cascade not found')

cap = cv2.VideoCapture(0)
ds_factor = 0.5

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=ds_factor, fy=ds_factor, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y, w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        # face detected
        cv2.rectangle(frame,(x,y), (x+w, y+h), (0,255,0), 3)
        eyes = eye_cascade.detectMultiScale(roi_gray)
        cv2.imshow('roi color', roi_color)
        # cv2.waitKey()
        for (x_eye, y_eye, w_eye, h_eye) in eyes:
            cv2.rectangle(roi_color, (x_eye, y_eye), (x_eye + w_eye, y_eye + h_eye), (255, 255, 0), 3)
            center = (int(x_eye + 0.5*w_eye), int(y_eye + 0.5*h_eye))
            radius = int(0.3 * (w_eye + h_eye))
            color = (0, 255, 0)
            thickness = 3
            cv2.circle(roi_color, center, radius, color, thickness)

    cv2.imshow('Eye Detector', frame)
    c = cv2.waitKey(1)
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()
