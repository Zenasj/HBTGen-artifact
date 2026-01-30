from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf
import numpy as np

input1 = tf.keras.layers.Input(shape=(128,128,1))
x = tf.keras.layers.BatchNormalization()(input1)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(5, activation = 'softmax')(x)

model = tf.keras.models.Model(input1, x)

batchSize = 64
x = np.random.rand(1, 128,128,1)
x = np.repeat(x, batchSize, axis=0)

print('The following rows should all be equal...')
for k in range(1, batchSize):  
    y = model.predict(x[0:k,:,:,:], batch_size=8)
    print(y[0,:])

import random
import tensorflow as tf
import numpy as np

tf.keras.backend.clear_session()
random.seed(1)
np.random.seed(1)
tf.random.set_random_seed(1) 
input1 = tf.keras.layers.Input(shape=(128,128,1))
#x = tf.keras.layers.BatchNormalization()(input1)
x = tf.keras.layers.Flatten()(input1)
x = tf.keras.layers.Dense(5, activation = 'softmax')(x)

model = tf.keras.models.Model(input1, x)

batchSize = 32
x = np.random.rand(1, 128,128,1)
x = np.repeat(x, batchSize, axis=0)

print('The following rows should all be equal...')
for k in range(1, batchSize):  
    y = model.predict(x[0:k,:,:,:], batch_size=8)
    print(y[0,:])