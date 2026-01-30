import random
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

import sys
import os.path
import tensorflow as tf
import math as m
import numpy as np
import scipy.io
from scipy.interpolate import griddata
import time
from itertools import product, combinations
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Layer, Activation, Dense, BatchNormalization
from tensorflow.keras.optimizers import Adam


inputs = tf.keras.Input(shape=(3,),dtype='float32')
layer1 = Dense(units=200, activation='tanh',dtype='float32')(inputs)
layer2 = Dense(units=200, activation='tanh',dtype='float32')(layer1)
layer3 = Dense(units=200, activation='tanh',dtype='float32')(layer2)
layer4 = Dense(units=200, activation='tanh',dtype='float32')(layer3)
layer5 = Dense(units=200, activation='tanh',dtype='float32')(layer4)
layer6 = Dense(units=200, activation='tanh',dtype='float32')(layer5)
layer7 = Dense(units=200, activation='tanh',dtype='float32')(layer6)
layer8 = Dense(units=200, activation='tanh',dtype='float32')(layer7)
layer9 = Dense(units=200, activation='tanh',dtype='float32')(layer8)
predictions = Dense(units=1, activation='linear',dtype='float32')(layer9)
model = tf.keras.Model(inputs=inputs, outputs=predictions)

# Define custom loss
def custom_loss():

    # Create a loss function
    def loss(y_true,y_pred):
        
        # Write here the loss function

        x0 = tf.random.uniform(shape=[2,100],maxval = 1)
        y0 = tf.random.uniform(shape=[2,100],maxval = 1)
        t0 = tf.random.uniform(shape=[100,2],maxval = 1)
        pY0 = tf.transpose(tf.concat([tf.ones([1,y0.shape[1]]) * y0[0,:], tf.zeros([1,y0.shape[1]], dtype='float32'), tf.ones([1,y0.shape[1]]) * y0[1,:]], 0))
        pY1 = tf.transpose(tf.concat([tf.ones([1,y0.shape[1]]) * y0[0,:], tf.ones([1,y0.shape[1]], dtype='float32'), tf.ones([1,y0.shape[1]]) * y0[1,:]], 0))
        pX0 = tf.transpose(tf.concat([tf.zeros([1,x0.shape[1]], dtype='float32'), tf.ones([1,x0.shape[1]]) * x0[0,:], tf.ones([1,x0.shape[1]]) * x0[1,:]], 0))
        pX1 = tf.transpose(tf.concat([tf.ones([1,x0.shape[1]], dtype='float32'), tf.ones([1,x0.shape[1]]) * x0[0,:], tf.ones([1,x0.shape[1]]) * x0[1,:]], 0))
        pIC = tf.concat([t0, tf.zeros([t0.shape[0],1], dtype='float32')], 1)
        pF = tf.random.uniform(shape=[1000,3],maxval = 1)
        uX0 = model(pX0, training=True)
        uX1 = model(pX1, training=True)
        uY0 = model(pY0, training=True)
        uY1 = model(pY1, training=True)
        uIC = model(pIC, training=True)
        u = model(pF, training=True)

        # compute the derivatives of u
        u_grad = tf.gradients(u, pF)[0]
        u_Dot = tf.gradients(u, pF)[0]
        u_t = u_grad[:,2]
        u_x = u_grad[:,0]
        u_y = u_grad[:,1]
        u_xx = (tf.gradients(u_x, pF))[0][:,0]
        u_yy = (tf.gradients(u_y, pF))[0][:,1]

        return tf.reduce_mean(tf.square(uX0)) + tf.reduce_mean(tf.square(uX1)) + \
        tf.reduce_mean(tf.square(uY0)) + tf.reduce_mean(tf.square(uY1)) + \
        tf.reduce_mean(tf.square(uIC - tf.transpose(tf.math.sin(pi*pIC[:,0])*tf.math.sin(pi*pIC[:,1]) * tf.ones([1,1], dtype='float32')) )) + \
        tf.reduce_mean(tf.square(u_t-u_xx-u_yy))

    # Return a function
    return loss

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4) ,
              loss=custom_loss(),
              # Call the loss function with the selected layer
              metrics=['accuracy'])

input_T = tf.linspace([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], 1000) # We define a dummy input in order to use the fit method
output_T = tf.linspace([0.0], [1.0], 1000) # We define a dummy output
model.fit(input_T, output_T,epochs=10, batch_size = input_T.shape[0])