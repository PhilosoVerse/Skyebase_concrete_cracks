import pickle
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense ,Dropout, Flatten, BatchNormalization
import matplotlib.pyplot as plt

################################################################################################

X = pickle.load(open('X_skyebase.pkl','rb'))
y = pickle.load(open('y_skyebase.pkl','rb'))

X = X/255 #feature scaling 0-1
print(X)
print(X.shape) # 8000 images , 100x100 pixels, 3 channels RGB

################################################################################################

model = Sequential()

model.add(Conv2D(64,(3,3),activation='relu')) # 3x3 matrix scanner , 64 feature extracters
model.add(MaxPool2D((2,2),))
model.add(Dropout(0.05))
model.add(BatchNormalization())

model.add(Conv2D(64,(3,3),activation='relu')) # 3x3 matrix scanner , 64 feature extracters
model.add(MaxPool2D((2,2),))
model.add(Dropout(0.05))
model.add(BatchNormalization())

model.add(Flatten())
model.add(Dense(128, input_shape=X.shape[1:0],activation='relu'))
model.add(Dropout(0.05))

model.add(Dense(2,activation='softmax'))

##################################################################################################

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

history = model.fit(X, y,batch_size=32, epochs=7, validation_split=0.2 , verbose=1)
####################################################################################################

import pickle

#with open('test_pickle','wb') as f:
#    pickle.dump(model,f)



####################
################
###########

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score
import pickle
import numpy as np
import os
import cv2
import random

#with open('test_picle','rb') as f:
#    model = pickle.load(f)
#    #model.predict(6)

####################################################

# Inladen van de dataset    y_test X_test

DIRECTORY = "C:\\Users\\Steven_Verkest\\Documents\\INF_STUDY\\BECODE\\PROJECT_SKYEBASE\\DATASET_KAGGLE"
CATEGORIES = ['Negative_test','Positive_test']

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

pickle.dump(X,open('X_test.pkl','wb'))
pickle.dump(y,open('y_test.pkl','wb'))

##################################################


y_pred = model.predict_classes(X)
print('\n')
print('accuracy score:', accuracy_score(y, y_pred) * 100)
print('\n')
print(classification_report(y, y_pred))
cf = confusion_matrix(y, y_pred)
print(cf)

