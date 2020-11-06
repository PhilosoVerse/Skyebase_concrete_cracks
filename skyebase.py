import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import itertools
import shutil
import random
import glob
import matplotlib.pyplot as plt
import os
import pandas as pd
import cv2
import pickle


######################################################################################################################

DIRECTORY = "C:\\Users\\Steven_Verkest\\Documents\\INF_STUDY\\BECODE\\PROJECT_SKYEBASE\\skyebase_dataset"
CATEGORIES = ['Negative','Positive']

IMG_SIZE =  100 #in order for CNN to be trained properly image size have to been similar across al the photos, play around with pixel size
data = []

for category in CATEGORIES:
    folder = os.path.join(DIRECTORY,category)
    label = CATEGORIES.index(category) #give label to
    for img in os.listdir(folder):
        img_path = os.path.join(folder,img)
        img_arr = cv2.imread(img_path)
        img_arr = cv2.resize(img_arr,(IMG_SIZE,IMG_SIZE)) #resize pixels so all image are same pixel_size
        data.append([img_arr, label])

print(len(data))

random.shuffle(data)

#print(data[0])

X = []
y = []

for features,labels in data:
    X.append(features)
    y.append(labels)

X = np.array(X)
y = np.array(y)
#print(len(X))

pickle.dump(X,open('X_skyebase.pkl','wb'))
pickle.dump(y,open('y_skyebase.pkl','wb'))
