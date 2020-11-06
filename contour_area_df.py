#here i make a dataframe of all concrete with cracks, crack surface area is measured with contourarea function of opencv

import numpy as np
import cv2
import re
from matplotlib import pyplot as plt
import pandas as pd
import os

DIRECTORY = "C:\\Users\\Steven_Verkest\\Documents\\INF_STUDY\\BECODE\\PROJECT_SKYEBASE\\DATASET_KAGGLE"
CATEGORIES = ['Negative','Positive']

IMG_SIZE =  227 #in order for CNN to be trained properly image size have to been similar across al the photos, play around with pixel size
data = []

df = pd.DataFrame()
#df['test'] = [1,2,3]
#print(df.head())

total_crack_area = []
images = []


for category in CATEGORIES:
    folder = os.path.join(DIRECTORY,category)
    label = CATEGORIES.index(category) #give label to
    for img in os.listdir(folder):
        img_path = os.path.join(folder,img)
        img_arr = cv2.imread(img_path)
        img_arr = cv2.resize(img_arr,(IMG_SIZE,IMG_SIZE)) #resize pixels so all image are same pixel_size
        data.append([img_arr, label])

        img = cv2.imread(img_arr)
        imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, thresh = cv2.threshold(imgray, 111, 255, cv2.THRESH_BINARY)
        contours, hierachy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        areas = []
        for contour in contours:
            area = cv2.contourArea(contour)
            areas.append(area)
            #print('area -> ', area)

        total_area = 0
        for x in areas:
            total_area += x
        print('total_Area ->', total_area)


        total_crack_area.append(total_area)
        images.append(img_path)

df['images'] = images
df['crack_surface'] = total_crack_area
df.to_csv('crack_surface.csv')

print(df.head())
