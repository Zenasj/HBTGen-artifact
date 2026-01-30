from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

import os
import sys
import random
import warnings

import numpy as np
import pandas as pd
import cv2

import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from PIL import ImageFile
import SimpleITK as sitk

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dropout, Lambda,Activation
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, BatchNormalization
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop,Adam,SGD,Adadelta


IMG_WIDTH = 512
IMG_HEIGHT = 512
IMG_CHANNELS = 3


DATA_PATH = 'G:/DenseUnet/input/test/'
np.random.seed = 42

sub_dir = 'image/'
image_ids =  next(os.walk(DATA_PATH + sub_dir))[2]

X = np.zeros((len(image_ids), IMG_HEIGHT, IMG_WIDTH,3))
Y = np.zeros((len(image_ids), IMG_HEIGHT, IMG_WIDTH,1))
mask = np.zeros((IMG_WIDTH, IMG_HEIGHT),dtype=np.bool)
for n, id_ in tqdm(enumerate(image_ids), total=len(image_ids)):
    path = DATA_PATH +sub_dir + id_
    img = sitk.ReadImage(path)
    img = sitk.GetArrayFromImage(img)
    #img = cv2.resize(img, (256, 256))
    X[n] = img
    
    mask = sitk.ReadImage(DATA_PATH + 'label/'+id_[:-8]+'gt.tiff')
    mask = sitk.GetArrayFromImage(mask)
    mask = np.expand_dims(mask, axis=2)
    Y[n] = mask

print('load files finished')
gc.collect()
X = (X-np.min(X))/(np.max(X)-np.min(X))
x_train = X
Y=Y.astype(bool)
y_train = Y

def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 4)
        tf.compat.v1.keras.backend.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return tf.keras.backend.mean(tf.keras.backend.stack(prec), axis=0)


# COMPETITION METRIC
def dice_coeff(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


#%% (Architecture)

def dense_block(inputs,growth_rate,n_layers):
    total_features=[]
    ini = inputs
    for i in range(n_layers):
        x = BatchNormalization()(ini)
        x = Activation('relu')(x)
        x = Conv2D(growth_rate, (3, 3),activation=None, kernel_initializer='he_normal', padding='same')(x)
        x = Dropout(0.2)(x)
        total_features.append(x)
        ini = concatenate([x,ini],axis=3)
        
        dense_out = total_features[0]
        for j in range(len(total_features)-1):
            dense_out = concatenate([dense_out,total_features[j+1]],axis=3)
        
    return dense_out,ini
       
def trans_down(inputs,filters):

    x = BatchNormalization()(inputs)
    x = Activation('relu')(x)
    
    x = Conv2D(filters, (1, 1), activation=None, kernel_initializer='he_normal', padding='same')(x)
    x = Dropout(0.2)(x)
    x = MaxPooling2D((2, 2))(x)
    
    return x

def trans_up(inputs,filters):
    x = Conv2DTranspose(filters, (3, 3), strides=(2, 2), padding='same')(inputs)
    return x

inputs = Input((IMG_HEIGHT, IMG_WIDTH,IMG_CHANNELS))
s = Lambda(lambda x: x / 1) (inputs)
c1 = Conv2D(48, (3, 3), activation=None, kernel_initializer='he_normal', padding='same')(s)
c1 = BatchNormalization()(c1)
c1 = Activation('relu')(c1)

b1,_ = dense_block(c1,16,4)
con1 = concatenate([b1,c1],axis=3)
d1 = trans_down(con1,112)

b2,_ = dense_block(d1,16,5)
con2 = concatenate([b2,d1],axis=3)
d2 = trans_down(con2,192)

b3,_ = dense_block(d2,16,7)
con3 = concatenate([b3,d2],axis=3)
d3 = trans_down(con3,304)

b4,_ = dense_block(d3,16,10)
con4 = concatenate([b4,d3],axis=3)
d4 = trans_down(con4,464)

b5,_ = dense_block(d4,16,12)
con5 = concatenate([b5,d4],axis=3)
d5 = trans_down(con5,656)

b6,block_to_up6 = dense_block(d5,16,15)

u7 = trans_up(block_to_up6,240)
con7 = concatenate([u7,con5],axis=3)
b7,block_to_up7 = dense_block(con7,16,12)

u8 = trans_up(block_to_up7,192)
con8 = concatenate([u8,con4],axis=3)
b8,block_to_up8 = dense_block(con8,16,10)

u9 = trans_up(block_to_up8,160)
con9 = concatenate([u9,con3],axis=3)
b9,block_to_up9 = dense_block(con9,16,7)


u10 = trans_up(block_to_up9,112)
con10 = concatenate([u10,con2],axis=3)
b10,block_to_up10 = dense_block(con10,16,5)


u11 = trans_up(block_to_up10,80)
con11 = concatenate([u11,con1],axis=3)
b11,block_to_up11 = dense_block(con11,16,4)
outputs = Conv2D(1, (1, 1), activation='sigmoid')(block_to_up11)

model = Model(inputs=[inputs], outputs=[outputs],name='DenseUNet')
model.compile(optimizer=RMSprop(1e-4),loss='binary_crossentropy', metrics=[dice_coeff])
model.summary()

filepath="G:/DenseUnet/model/20210812/0812-{epoch:02d}-{val_dice_coeff:.4f}.h5" 
earlystopper = EarlyStopping(patience=350, verbose=1)
checkpointer = ModelCheckpoint(filepath,monitor='val_loss', verbose=1, save_best_only=False)
results = model.fit(x_train, y_train, validation_split=0.2, batch_size=3, epochs=100, 
                    callbacks=[earlystopper,checkpointer])

import pickle
with open('G:/DenseUnet/0810_100e.txt', 'wb') as file_txt:
    pickle.dump(results.history, file_txt)