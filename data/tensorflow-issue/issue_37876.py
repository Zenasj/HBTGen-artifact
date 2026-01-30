import random
from tensorflow.keras import optimizers

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

num_inputs=1440*3
num_outputs=2

N = 1000*1000
train_X = np.random.randn(N, num_inputs)
train_y = np.random.randn(N, num_outputs)

tf.keras.backend.set_floatx('float32')
model = keras.Sequential()
n_hidden = 2*num_inputs

activation1 = tf.nn.relu
model.add(layers.Dense(n_hidden, activation=activation1, input_shape=(num_inputs,)))
for i in range(4):
    model.add(layers.Dense(n_hidden, activation=activation1))
model.add(layers.Dense(num_outputs))

model.compile(optimizer=tf.keras.optimizers.RMSprop(1e-3),
              loss=tf.keras.losses.mse,
              metrics=(tf.keras.metrics.mse,))

model.fit(train_X, train_y, epochs=1000, batch_size=512, verbose=2)

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

warnings.warn(
                'Method (%s) is slow compared '
                'to the batch update (%f). Check your callbacks.'
                % (hook_name, delta_t_median), RuntimeWarning)

tbCallBack = TensorBoard(log_dir='logs', update_freq='epoch', write_graph=True, profile_batch=0)

import os
# To disable all logging output from TensorFlow
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"