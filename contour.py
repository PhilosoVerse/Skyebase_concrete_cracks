# in this file i experimented with trying to draw a contour around the crack
import numpy as np
import cv2
import re
from matplotlib import pyplot as plt

img = cv2.imread('00007.jpg')
imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

ret, thresh = cv2.threshold(imgray, 111,255,cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) #counter  = python list of all the contours in the image, each individual counter is numpy array with x,y coordinates
print("number of contours = ", str(len(contours)))
#print(contours[3]) # if contours less then 10 x&y coordinates -> bad contour

cv2.drawContours(img,contours, 3, (0,255,0),3) #-1 = all counters  //
cv2.drawContours(img,contours, 6, (0,255,0),3) #-1 = all counters  //
# 3 6

cv2.imshow('Image',img)
#cv2.imshow('Image GRAY', imgray)
cv2.waitKey(0)
cv2.destroyAllWindows()

#################################################################################################################

coordinates_contour = []

for coordinates in contours[6]:
    print(coordinates[0])   # 36 0 ...
    txt = str(coordinates[0])
    coordinates_contour.append(txt)


x_values_contour = []
y_values_contour = []
for elements in coordinates_contour:
    split_elemnts = elements.split()
    x = re.findall("[0-9]", split_elemnts[0])
    x_values_contour.append(x)
    #print(x)
    #print('-')
    y = re.findall("[0-9]", split_elemnts[1])
    y_values_contour.append(y)
    #print(y)
    #print('###')



x_coordinates = []

for elements in x_values_contour:
    length = len(elements)
    if(length ==3):
       x = elements[0] + elements[1] + elements[2]
       x = int(x)
       #print(x)
       x_coordinates.append(x)
    if (length == 2):
        x = elements[0] + elements[1]
        x = int(x)
        #print(x)
        x_coordinates.append(x)
    if (length == 1):
        x = elements[0]
        x = int(x)
        #print(x)
        x_coordinates.append(x)


print('######################################')

y_coordinates = []

for elements in y_values_contour:
    length = len(elements)
    if(length == 3):
       y = elements[0] + elements[1] + elements[2]
       y = int(y)
       #print(y)
       y_coordinates.append(y)
    if (length == 2):
        y = elements[0] + elements[1]
        y = int(y)
        #print(y)
        y_coordinates.append(y)
    if (length == 1):
        y = elements[0]
        y = int(y)
        #print(y)
        y_coordinates.append(y)

# Xmin Ymax  |   Xmax Ymax
# Xmin Ymin  |   Xmax Ymin

X_max = max(x_coordinates)
Y_max = max(y_coordinates)
X_min = min(x_coordinates)
Y_min = min(y_coordinates)

print('#################')
print(X_max)
print(Y_max)
print(X_min)
print(Y_min)
print('#################')

# 36,226   107,226      (6)
# 36, 0    107,0        (6)

# 0,226   107, 226      (3)
# 0,0     107,0         (3)

#subtract pixel values from contour 3 & contour 6









