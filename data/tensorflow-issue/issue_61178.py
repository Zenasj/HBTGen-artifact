import random
from tensorflow.keras import layers
from tensorflow.keras import models

# import tensorflow as tf
# import tensorflow.keras
import keras
import tensorflow as tf
import tensorflow.keras as k2

print("CPU LIST:", tf.config.list_physical_devices("CPU"))
print("GPU LIST:", tf.config.list_physical_devices("GPU"))
print("Deprecated AVAILABLE:", tf.test.is_gpu_available())  # Deprecated
print("Deprecated AVAILABLE:", tf.test.is_gpu_available(cuda_only=False))  # Deprecated
print("BUILD WITH CUDA:", tf.test.is_built_with_cuda())  # Installed non gpu package

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM, Flatten
from tensorflow.keras.layers import ConvLSTM2D

import numpy as np

import keras


# tf.compat.v1.InteractiveSession() #3-4ms
# with tf.compat.v1.Session():
# None

N = int(3e4)
X = np.random.random((N, 20))
Y = np.random.random(N)

#######
"Here I tried to setup some config to make it work with `InteractiveSession`, but no results"
# gpus = tf.config.experimental.list_physical_devices('GPU')
# gpu_conf = tf.config.experimental.set_virtual_device_configuration(
#         gpus[0],
#         [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4000)])
# logical_gpus = tf.config.experimental.list_logical_devices('GPU')
# print(f"Logical: {logical_gpus}")
# 
# config = tf.compat.v1.ConfigProto(gpu_options=gpu_conf)
session = tf.compat.v1.InteractiveSession()

####################
"Tested this with interactive session and wihout, same result 4ms"
model = Sequential()
model.add(Dense(50, input_shape=(20,)))
model.add(Dense(60))
model.add(Dense(60))
model.add(Dense(60))
model.add(Dense(60))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

model.fit(X, Y, verbose=True, epochs=1)
model.predict(X)

####################
"Session 70-110us which is notable difference"
with tf.compat.v1.Session():
    model = Sequential()
    model.add(Dense(50, input_shape=(20,)))
    model.add(Dense(60))
    model.add(Dense(60))
    model.add(Dense(60))
    model.add(Dense(60))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

    model.fit(X, Y, verbose=True, epochs=1)

    model.predict(X)