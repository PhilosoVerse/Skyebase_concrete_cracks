#in this file a try to draw a rectagnle box around the crack
import numpy as np
import cv2

img = cv2.imread('00293.jpg',0)
# 36,226   107,226      (6)
# 36, 0    107,0        (6)
# 0,226   107, 226      (3)
# 0,0     107,0         (3)

#img = cv2.line(img, (0,0), (255,255),(255,0,0),3) #BGR  | thickness
img = cv2.rectangle(img, (0,100),(225,200),(0,255,255),3)
img = cv2.putText(img,'crack',(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3) #coordinates you start text from, font,fontsize,thickness
img = cv2.putText(img,'100x225pxls',(25,25),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),1)

cv2.imshow('crack',img)

cv2.waitKey(0)
cv2.destroyAllWindows()

