from tensorflow import keras

import tensorflow as tf
import numpy as np

class MyModel(tf.keras.Model):    
    def call(self, inputs, training):
        # make a model with a batched input (x) and per-batch tensor (s)
        x = inputs[0]
        s = inputs[1]
        return tf.reduce_mean(x * s, axis=1)
    
m = MyModel()
x = np.ones((10, 2))
s = np.ones(1)
y = np.ones(10)

m.compile('sgd', loss='mean_squared_error')

# works as call
m([x, s])

# fails on TF 2.4.1
# succeeds on TF 2.3.2
m.train_on_batch(x=[x, s], y=y)

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf