import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1' #using tensor cores
policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
tf.keras.mixed_precision.experimental.set_policy(policy)

encoder.add(Dense(eneurons[0],input_dim=dim,dtype=tf.float16))

if noise:
        encoder.add(GaussianNoise(noise))

import os
import tensorflow as tf
from tensorflow.keras.layers import Dense,GaussianNoise
from tensorflow.keras.models import Sequential


os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1' #using tensor cores
policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
tf.keras.mixed_precision.experimental.set_policy(policy)


#models
encoder = Sequential()
encoder.add(Dense(5,input_dim=5))
encoder.add(GaussianNoise(0.01))

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense,GaussianNoise,Activation
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from sklearn.metrics import mean_squared_error as mse


os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1' #using tensor cores
policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
tf.keras.mixed_precision.experimental.set_policy(policy)


encoder = Sequential()
encoder.add(Dense(5,input_dim=5,dtype=tf.float16))
encoder.add(GaussianNoise(0.01,dtype='float32'))
encoder.add(Activation('relu'))
encoder.add(Dense(5))    
    

input_dim = Input(shape = (5, ),name='rating_in')

model = Model(input_dim, encoder(input_dim))
    
model.compile(loss=tf.keras.losses.mean_squared_error)
  
data = np.asarray(np.random.rand(20,5),dtype='float16') ######## cast to float16
model.fit(data,data,batch_size=10)

def ModelTrain(model,optim,x,w,y,train_loss):
        with tf.GradientTape() as tape:
            predictions = model(x) 
            loss = MMSE(y,w,predictions)
            scaled_loss = optim.get_scaled_loss(loss)
        
        scaled_gradients = tape.gradient(scaled_loss, model.trainable_variables)
        gradients = optim.get_unscaled_gradients([tf.cast(x,tf.float32) for x in scaled_gradients])
        optim.apply_gradients(zip((gradients), model.trainable_variables))