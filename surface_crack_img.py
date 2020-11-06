#here i calculate the crack-surface area with the contourArea function (not saved in a dataframe)

import cv2

img = cv2.imread('00007.jpg')
imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ret, thresh = cv2.threshold(imgray, 111, 255, cv2.THRESH_BINARY)
contours, hierachy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

areas = []
for contour in contours:
    area = cv2.contourArea(contour)
    areas.append(area)
    print('area -> ', area)
