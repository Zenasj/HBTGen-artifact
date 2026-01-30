import random
from tensorflow.keras import models

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class RNNCellWithConstants(keras.layers.Layer):

  def __init__(self, units, constant_size, **kwargs):
    self.units = units
    self.state_size = units
    self.constant_size = constant_size
    super(RNNCellWithConstants, self).__init__(**kwargs)

  def build(self, input_shape):
    self.input_kernel = self.add_weight(
        shape=(input_shape[0][1], self.units),
        initializer='uniform',
        name='kernel')
    self.recurrent_kernel = self.add_weight(
        shape=(self.units, self.units),
        initializer='uniform',
        name='recurrent_kernel')
    self.constant_kernel = self.add_weight(
        shape=(self.constant_size, self.units),
        initializer='uniform',
        name='constant_kernel')
    self.built = True

  def call(self, inputs, states, constants):
    [x1, _] = inputs
    [prev_output] = states
    [constant] = constants
    h_input = keras.backend.dot(x1, self.input_kernel)
    h_state = keras.backend.dot(prev_output, self.recurrent_kernel)
    h_const = keras.backend.dot(constant, self.constant_kernel)
    output = h_input + h_state + h_const
    return output, [output]

  def get_config(self):
    config = {'units': self.units, 'constant_size': self.constant_size}
    base_config = super(RNNCellWithConstants, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


x1 = keras.Input((None, 5))
x2 = keras.Input((None, 5))
c = keras.Input((3,))
cell = RNNCellWithConstants(32, constant_size=3)
layer = keras.layers.RNN(cell)
y = layer((x1,x2), constants=c)

model = keras.models.Model([x1, x2, c], y)
model.compile(
    optimizer='rmsprop',
    loss='mse')
model.train_on_batch(
    [np.zeros((6, 5, 5)), np.zeros((6, 5, 5)), np.zeros((6, 3))],
    np.zeros((6, 32))
)

# Test basic case 
x1_np = np.random.random((6, 5, 5))
x2_np = np.random.random((6, 5, 5))
c_np = np.random.random((6, 3))
y_np = model.predict([x1_np, x2_np, c_np]) 

model.save("test.h5")
loaded_model = keras.models.load_model("test.h5", custom_objects={"RNNCellWithConstants":RNNCellWithConstants})
loaded_y_np = loaded_model.predict([x1_np, x2_np, c_np]) 
assert np.array_equal(y_np, loaded_y_np)