from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import glob
import cv2
import os
import glob
import tensorflow as tf
import pickle
from scipy import ndimage as nd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow .keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import sklearn.utils 
from tensorflow.keras.optimizers import Adadelta, RMSprop,SGD,Adam

print(tf.__version__)

train_path="C://Users//Tusneem//Documents//Untitled Folder//Tusneem_Seg//SEG_OBJ1//BAS//ROI_1//"
mask_path="C://Users//Tusneem//Documents//Untitled Folder//Tusneem_Seg//SEG_OBJ1//BAS//ROI_2//"

############################################################################
size = 231 
images = []

for directory_path in glob.glob("C://Users//Tusneem//Documents//Untitled Folder//Tusneem_Seg//SEG_OBJ1//BAS//ROI_1//ROI_train//"):
    for img_path in glob.glob(os.path.join(directory_path, "*.tiff")):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)       
        img = cv2.resize(img, (size, size))
        #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        images.append(img)
        #train_labels.append(label)
        
images = np.array(images)

masks = [] 
for directory_path in glob.glob("C://Users//Tusneem//Documents//Untitled Folder//Tusneem_Seg//SEG_OBJ1//BAS//ROI_2//ROI_train//"):
    for mask_path in glob.glob(os.path.join(directory_path, "*.tiff")):
        mask = cv2.imread(mask_path, 0)       
        mask = cv2.resize(mask, (size, size))
        masks.append(mask)
                
masks = np.array(masks)
masks=np.expand_dims(masks, axis=3)

#######################Example of transforming images and masks together.
import tensorflow as tf

# we create two instances with the same arguments
data_gen_args = dict(featurewise_center=True,
                     featurewise_std_normalization=True,
                     rotation_range=90)
                     
## what is images
## what is masks
# what is **data_gen_args

image_datagen = ImageDataGenerator(data_gen_args)
mask_datagen = ImageDataGenerator(data_gen_args)
# Provide the same seed and keyword arguments to the fit and flow methods
seed = 1
image_datagen.fit(images, augment=True, seed=seed)
mask_datagen.fit(masks, augment=True, seed=seed)

image_generator = image_datagen.flow_from_directory(
    train_path,
    class_mode=None,
    shuffle= False,
    seed=seed)

mask_generator = mask_datagen.flow_from_directory(
    "C://Users//Tusneem//Documents//Untitled Folder//Tusneem_Seg//SEG_OBJ1//BAS//ROI_2//",
    class_mode=None,
    shuffle= False,
    seed=seed)
# combine generators into one which yields image and masks
#train_generator = zip(image_generator, mask_generator)
train_generator = (pair for pair in zip(image_generator, mask_generator))

activation= 'relu'

model = Sequential()
model.add(Conv2D(16, kernel_size=(3, 3),activation='relu',input_shape=(231,231,3),padding='same'))
model.add(Conv2D(16, kernel_size=(3, 3),activation='relu',padding='same'))
model.add(Flatten())
model.add(Dense(5))
model.add(Dense(3))
model.compile( loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()

model.fit_generator(train_generator, steps_per_epoch= 15, epochs=2, verbose=1)