import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf
import numpy as np

def Model():
    
    x = tf.keras.layers.Input((4,4,3)) # 4x4 image with 3 channels
    y = tf.keras.layers.Conv2DTranspose(3, 4, 2, padding='same') (x) # Creates a 8x8 images with 3 channels by upsampling with stride 2
    
    linear = tf.keras.layers.Dense(8*8*3) (x) # Linear transformation of the input
    linear = tf.keras.layers.Reshape([8, 8, 3]) (linear) # Reshapes the output of the linear transformation to the same shape as the upsampled image
    y = tf.keras.layers.Add() ([y, linear]) # adds them together
    
    return tf.keras.models.Model(inputs=x, outputs=y)

# model
model = Model()

# input data
array_range = np.random.randn(128, 4, 4,  3).astype(np.float32)
dataset = tf.data.Dataset.from_tensor_slices(array_range).batch(8)
iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()
dataset_output = model(next_element)

# session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# evaluate
print(sess.run(dataset_output))
print(sess.run(dataset_output))

import tensorflow as tf
import numpy as np

def Model():
    
    x = tf.keras.layers.Input((4,4,3)) # 4x4 image with 3 channels
    y = tf.keras.layers.Conv2DTranspose(3, 4, 2, padding='same') (x) # Creates a 8x8 images with 3 channels by upsampling with stride 2
    
    linear = tf.keras.layers.Flatten() (x)
    linear = tf.keras.layers.Dense(8*8*3) (linear) # Linear transformation of the input
    linear = tf.keras.layers.Reshape([8, 8, 3]) (linear) # Reshapes the output of the linear transformation to the same shape as the upsampled image
    y = tf.keras.layers.Add() ([y, linear]) # adds them together
    
    return tf.keras.models.Model(inputs=x, outputs=y)

# model
model = Model()

# input data
array_range = np.random.randn(128, 4, 4,  3).astype(np.float32)
dataset = tf.data.Dataset.from_tensor_slices(array_range).batch(8)
iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()
dataset_output = model(next_element)

# session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# evaluate
print(sess.run(dataset_output))
print(sess.run(dataset_output))