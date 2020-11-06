#detecting cracks using the camera of the pc in real time

import cv2
import numpy as np
cap = cv2.VideoCapture(0)

while(True):
    _, frame = cap.read()
    blurred_frame = cv2.GaussianBlur(frame,(5,5),0)
    hsv = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)

    lower_grey = np.array([0,0,0])
    upper_grey = np.array([111,111,111])
    mask = cv2.inRange(hsv,lower_grey,upper_grey)

    contours, hierachy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)

        if area > 1000:
            cv2.drawContours(frame,contour,-1, (0,255,0),3)

    #print(contours)
    cv2.drawContours(frame, contours, -1,(0, 255, 0),3)

    cv2.imshow('frame',frame)
    cv2.imshow('mask',mask)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()