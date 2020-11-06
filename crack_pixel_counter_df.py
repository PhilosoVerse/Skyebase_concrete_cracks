#here i make a dataframe of all concrete with cracks, crack surface area is measured counting the pixels
# between a certain RGB treshholdvalue
import cv2
from matplotlib import pyplot as plt
import os
import pandas as pd

DIRECTORY = "C:\\Users\\Steven_Verkest\\Documents\\INF_STUDY\\BECODE\\PROJECT_SKYEBASE\\DATASET_KAGGLE"
CATEGORIES = ['Negative','Positive']

IMG_SIZE =  227 #in order for CNN to be trained properly image size have to been similar across al the photos, play around with pixel size
data = []

df = pd.DataFrame()
#df['test'] = [1,2,3]
#print(df.head())

crack_pixel_size_array = []
images = []


for category in CATEGORIES:
    folder = os.path.join(DIRECTORY,category)
    label = CATEGORIES.index(category) #give label to
    for img in os.listdir(folder):
        img_path = os.path.join(folder,img)
        img_arr = cv2.imread(img_path)
        img_arr = cv2.resize(img_arr,(IMG_SIZE,IMG_SIZE)) #resize pixels so all image are same pixel_size
        data.append([img_arr, label])

        pixel_counter = 0

        for x in range(0, 227):
            px = img_arr[x, x]

            if min(px) < 111:
                pixel_counter += 1

        print("crack pixels ->", pixel_counter)
        crack_pixel_size_array.append(pixel_counter)
        images.append(img_path)

df['images'] = images
df['crack_pixels_size'] = crack_pixel_size_array
df.to_csv('crack_pixels.csv')

print(df.head())




