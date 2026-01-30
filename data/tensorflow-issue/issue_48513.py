from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import tensorflow as tf
model1 = Unet(input_shape = (129,239,1))
model1.load_weights("model/variables/variables")

model2= Unet(input_shape = (2300,3450,1))
model2.load_weights("model/variables/variables")

prediction1 = model1(input_tensor1) # input_tensor1 shape is (None, 129,239,1), None is the batch size.
prediction1 = model2(input_tensor2) # input_tensor2 shape is (None, 2300,3450,1), None is the batch size.

import os
import skimage.io as io
import cv2
import numpy as np
import tensorflow as tf
import time
import imageio
import matplotlib.pyplot as plt
import copy as copy
import matplotlib

def get_crop_shape(target, query):
	# the height
	channelHeight = target.get_shape()[1] - query.get_shape()[1]
	assert (channelHeight >= 0)
	channelHeight1 = int(channelHeight/2)
	if channelHeight % 2 != 0:
		channelHeight2 = channelHeight1 + 1
	else:
		channelHeight2 = channelHeight1
	# the width
	channelWidth = target.get_shape()[2] - query.get_shape()[2]
	assert (channelWidth >= 0)
	channelWidth1 = int(channelWidth/2)
	if channelWidth % 2 != 0:
		channelWidth2 = channelWidth1 + 1
	else:
		channelWidth2 = channelWidth1
	return (channelHeight1, channelHeight2), (channelWidth1, channelWidth2)


def getAct(x):
	return tf.keras.layers.ReLU()(x)


def Unet(input_shape = (None,None,1), kernelSize = 3, drop_level = 0.5, nChannels = 1):
	inputs = tf.keras.layers.Input(shape = [input_shape[0], input_shape[1], input_shape[2]])
	conv1 = tf.keras.layers.Conv2D(64, kernelSize, padding = 'same', kernel_initializer = 'he_normal')(inputs)
	conv1 =  getAct(conv1)
	conv1 = tf.keras.layers.Conv2D(64, kernelSize, padding = 'same', kernel_initializer = 'he_normal')(conv1)
	conv1 =  getAct(conv1)
	#drop1 = tf.keras.layers.Dropout(drop_level)(conv1)
	pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1) # drop1 --> conv1
	#
	conv2 = tf.keras.layers.Conv2D(128, kernelSize, padding = 'same', kernel_initializer = 'he_normal')(pool1)
	conv2 =  getAct(conv2)
	conv2 = tf.keras.layers.Conv2D(128, kernelSize, padding = 'same', kernel_initializer = 'he_normal')(conv2)
	conv2 =  getAct(conv2)
	#drop2 = tf.keras.layers.Dropout(drop_level)(conv2)
	pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2) #drop2 --> conv2
	#
	conv3 = tf.keras.layers.Conv2D(256, kernelSize, padding = 'same', kernel_initializer = 'he_normal')(pool2)
	conv3 =  getAct(conv3)
	conv3 = tf.keras.layers.Conv2D(256, kernelSize, padding = 'same', kernel_initializer = 'he_normal')(conv3)
	conv3 =  getAct(conv3)
	#drop3 = tf.keras.layers.Dropout(drop_level)(conv3)
	pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)  #drop3 --> conv3
	#
	conv4 = tf.keras.layers.Conv2D(512, kernelSize, padding = 'same', kernel_initializer = 'he_normal')(pool3)
	conv4 =  getAct(conv4)
	conv4 = tf.keras.layers.Conv2D(512, kernelSize, padding = 'same', kernel_initializer = 'he_normal')(conv4)
	conv4 =  getAct(conv4)
	drop4 = tf.keras.layers.Dropout(drop_level)(conv4)
	pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(drop4)
	#
	conv5 = tf.keras.layers.Conv2D(1024, kernelSize, padding = 'same', kernel_initializer = 'he_normal')(pool4)
	conv5 =  getAct(conv5)
	conv5 = tf.keras.layers.Conv2D(1024, kernelSize, padding = 'same', kernel_initializer = 'he_normal')(conv5)
	conv5 =  getAct(conv5)
	drop5 = tf.keras.layers.Dropout(drop_level)(conv5)
	up6 = tf.keras.layers.Conv2DTranspose(512, kernelSize, strides = (2,2), padding = 'same', kernel_initializer = 'he_normal')(drop5)
	ch, cw = get_crop_shape(drop4, up6)
	up6 = tf.keras.layers.ZeroPadding2D(padding=((ch[0], ch[1]), (cw[0], cw[1])))(up6) # add zeropadding.
	merge6 = tf.keras.layers.concatenate([drop4,up6], axis = 3)
	#
	conv6 = tf.keras.layers.Conv2D(512, kernelSize, padding = 'same', kernel_initializer = 'he_normal')(merge6)
	conv6 =  getAct(conv6)
	conv6 = tf.keras.layers.Conv2D(512, kernelSize, padding = 'same', kernel_initializer = 'he_normal')(conv6)
	conv6 =  getAct(conv6)
	up7 = tf.keras.layers.Conv2DTranspose(256, kernelSize, strides = (2,2), padding = 'same', kernel_initializer = 'he_normal')(conv6)
	ch, cw = get_crop_shape(conv3, up7)   #drop3 --> conv3
	up7 = tf.keras.layers.ZeroPadding2D(padding=((ch[0], ch[1]), (cw[0], cw[1])))(up7) # add zeropadding.
	merge7 = tf.keras.layers.concatenate([conv3,up7], axis = 3)   #drop3 --> conv3
	#
	conv7 = tf.keras.layers.Conv2D(256, kernelSize, padding = 'same', kernel_initializer = 'he_normal')(merge7)
	conv7 =  getAct(conv7)
	conv7 = tf.keras.layers.Conv2D(256, kernelSize, padding = 'same', kernel_initializer = 'he_normal')(conv7)
	conv7 =  getAct(conv7)
	up8 = tf.keras.layers.Conv2DTranspose(128, kernelSize, strides = (2,2), padding = 'same', kernel_initializer = 'he_normal')(conv7)
	ch, cw = get_crop_shape(conv2, up8) #drop2 --> conv2
	up8 = tf.keras.layers.ZeroPadding2D(padding=((ch[0], ch[1]), (cw[0], cw[1])))(up8) # add zeropadding.
	merge8 = tf.keras.layers.concatenate([conv2,up8], axis = 3) #drop2 --> conv2
	#
	conv8 = tf.keras.layers.Conv2D(128, kernelSize, padding = 'same', kernel_initializer = 'he_normal')(merge8)
	conv8 =  getAct(conv8)
	conv8 = tf.keras.layers.Conv2D(128, kernelSize, padding = 'same', kernel_initializer = 'he_normal')(conv8)
	conv8 =  getAct(conv8)
	up9 = tf.keras.layers.Conv2DTranspose(64, kernelSize, strides = (2,2), padding = 'same', kernel_initializer = 'he_normal')(conv8)
	ch, cw = get_crop_shape(conv1, up9) #drop1 --> conv1
	up9 = tf.keras.layers.ZeroPadding2D(padding=((ch[0], ch[1]), (cw[0], cw[1])))(up9) # add zeropadding.
	merge9 = tf.keras.layers.concatenate([conv1,up9], axis = 3) #drop1 --> conv1
	#
	conv9 = tf.keras.layers.Conv2D(64, kernelSize, padding = 'same', kernel_initializer = 'he_normal')(merge9)
	conv9 =  getAct(conv9)
	conv9 = tf.keras.layers.Conv2D(64, kernelSize, padding = 'same', kernel_initializer = 'he_normal')(conv9)
	conv9 =  getAct(conv9)
	conv9 = tf.keras.layers.Conv2D(2, kernelSize, padding = 'same', kernel_initializer = 'he_normal')(conv9)
	conv9 =  getAct(conv9)
	conv10 = tf.keras.layers.Conv2D(nChannels, 1, activation = 'sigmoid')(conv9)
	model = tf.keras.Model(inputs = inputs, outputs = conv10)
	model.compile(optimizer = tf.keras.optimizers.Adam(1e-4, beta_1 = 0.9, beta_2 = 0.999), loss = 'binary_crossentropy', metrics = ['accuracy'])
	print("Using UnetLeakyPReLU ...")
	print(model.summary())
	return model

model1 = Unet(input_shape = (129,239,1))
model1.load_weights("model/variables/variables")

import tensorflow as tf
model1 = Unet(input_shape = (256,256,1))
model1.load_weights("model/variables/variables")