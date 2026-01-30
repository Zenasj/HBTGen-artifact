import random
from tensorflow import keras
from tensorflow.keras import layers

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np

input = np.random.rand(2, 8, 8, 8)
x = tf.keras.Input([None, None, 8])
y = tf.keras.layers.Conv2DTranspose(filters=0,kernel_size=3, padding='same', dilation_rate=(1,1))(x)
model = tf.keras.Model(x, y)
z = model(input).numpy()
print(z.mean())

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np

input = np.random.rand(2, 8, 8, 8)
x = tf.keras.Input([None, None, 8])

y = tf.keras.layers.Conv2D(0, kernel_size=3)(x)
model = tf.keras.Model(x, y)
z = model(input).numpy()
print(z.shape)

Conv1DTranspose
Conv3DTranspose
ConvLSTM2D

Conv1D
Conv3D