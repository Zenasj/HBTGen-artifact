import random
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
import numpy as np

x = np.random.rand(1, 10)
y = np.random.rand(4)

my_model = tf.keras.Sequential(layers=tf.keras.layers.Dense(4, input_shape=(10,) ))
my_model.compile(loss='mse', target_tensors=[y], run_eagerly=False)

my_model.fit(x)

import tensorflow as tf
import numpy as np

x = np.random.rand(1, 10)
y = np.random.rand(4)

my_model = tf.keras.Sequential(layers=tf.keras.layers.Dense(4, input_shape=(10,) ))
my_model.compile(loss='mse', target_tensors=[y], run_eagerly=False)

for _ in range(epochs):
    my_model.evaluate(x, y = None)

for _ in range(epochs):
    my_model.evaluate(x, y = y) # very slow

def make_test_function(self):
        @tf.function
        def evaluate():
            ret = calculate_loss_of_model()
            return ret
        return evaluate