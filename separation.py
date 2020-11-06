#here i try to do image segmentation based on using a certain RGB treshold value
import cv2
from matplotlib import pyplot as plt

img = cv2.imread("00007.jpg")
#img_color = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ret, tresh = cv2.threshold(gray,111,255,cv2.THRESH_BINARY) #values between 128 & 255 are background , white ore black
#ret, tresh2 = cv2.threshold(gray,111,255,cv2.THRESH_BINARY+ cv2.THRESH_OTSU)
#ret,tresh3 = cv2.threshold(gray,111,255,cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)

print(ret) # best black-white value

plt.figure("tresh")
plt.imshow(tresh,cmap='gray')

plt.show()