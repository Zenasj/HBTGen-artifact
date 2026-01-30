import math
import random
from tensorflow.keras import layers
from tensorflow.keras import models

from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras import *
import tensorflow as tf
import numpy as np

strat = tf.distribute.MirroredStrategy(devices=['/gpu:0', '/gpu:1'])

with strat.scope():
    model = Sequential()
    model.add(MaxPooling3D((1, 2, 2), input_shape=(None, 640, 480, 1)))
    model.add(ConvLSTM2D(16, kernel_size=(3,3)))
    model.add(Flatten())
    model.add(Dense(1))
    model.summary()
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['acc'])

    x = np.random.rand(1, 10, 640, 480, 1)
    y = np.random.rand(1, 1)
    model.fit(x=x, y=y, epochs=10)

# Assuming `ds` has 89 records in total
num_records = 89
batch_size = 8
ds = ds.batch(batch_size).repeat().cache().prefetch(tf.data.experimental.AUTOTUNE) # i.e., 11 batches of 8 records, and 1 batch of 1 record
step_size = math.ceil(num_records / batch_size) # i.e., 12
...
model.fit(
  ds,
  steps_per_epoch=step_size,
  # other arguments
  )