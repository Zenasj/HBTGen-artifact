import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf
import numpy as np

use_eager     = True
use_TFDataset = True

if not use_eager:
    tf.compat.v1.disable_eager_execution()

# Build dataset
n_data = 10**5
my_data = np.random.random((n_data,10,1))
my_targets = np.random.randint(0,2,(n_data,1))
data = ({'x_input':my_data}, {'target':my_targets})

# Create tf.data.Dataset
BATCH_SIZE = 10
dataset = tf.data.Dataset.from_tensor_slices(data)
dataset = dataset.batch(BATCH_SIZE)
dataset = dataset.prefetch(1)

#Build model
x_input = tf.keras.layers.Input((None,1), name='x_input')
RNN = tf.keras.layers.SimpleRNN(100, name='RNN')(x_input)
hidden = tf.keras.layers.Dense(100, name='hidden')(RNN)
dense = tf.keras.layers.Dense(1, name='target')(hidden)
my_model = tf.keras.models.Model(inputs = [x_input], outputs = [dense])
my_model.compile(optimizer='SGD', loss = 'binary_crossentropy')

# Train model
if use_TFDataset:
    my_model.fit(dataset, epochs = 1, steps_per_epoch=n_data//BATCH_SIZE) # divide by BATCH_SIZE to keep the number of training steps the same
else:
    my_model.fit(x = my_data, y = my_targets, epochs = 1, batch_size= BATCH_SIZE)