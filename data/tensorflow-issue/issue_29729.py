import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

model = Sequential()
model.add(layers.Dense(32, input_shape=(500,)))
model.add(layers.Dense(32))
print(len(model.weights)) #output 4

class MyModel(tf.keras.Model):
  def __init__(self, state_dim, action_dim):
    super(MyModel, self).__init__()
    self.state_dim = state_dim
    self.action_dim = action_dim

    self.fc1 = tf.keras.layers.Dense(100, input_shape=(2,))
    self.fc2 = tf.keras.layers.Dense(1)

  def call(self, inputs):
    out = self.fc1(inputs)
    out = self.fc2(out)
    return out

Q = MyModel(2, 2)
print(len(Q.weights)) # output 0
states = np.random.random((10, 2))
Q(states)
print(len(Q.weights)) # output 4