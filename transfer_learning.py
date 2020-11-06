#exploring to use an pretrained CNN
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input, decode_predictions
from keras.preprocessing import image
from keras.utils import np_utils
import keras

modelVGG19 = keras.applications.vgg19.VGG19(include_top=False, weights='imagenet', input_shape=(64,64,3))
type(modelVGG19)

############################

# Omzetten naar een sequential model behalve output layer

model = Sequential()

for layer in modelVGG19.layers[:]:
    model.add(layer)

model.summary()

##########################

# alle layers vast zetten

for layer in model.layers:
    layer.trainable = False

# Toevoegen van de output layer met softmax activatiefunctie
model.add(Flatten())
model.add(Dense(50,activation='softmax'))
model.add(Dense(10,activation='softmax'))

#############################################

#sgd = keras.optimizers.SGD(lr=0.001,nesterov=True)
adam = keras.optimizers.adam()
model.compile(loss='categorical_crossentropy',optimizer =adam,metrics=['accuracy'])
model.summary()

###############################










