import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models

import TensorFlow as tf

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10)
])

print(model.inputs)
print(model.outputs)

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

class MyModel(Model):
  def __init__(self):
    super(MyModel, self).__init__()
    self.flatten = Flatten(input_shape=(28, 28, 1))
    self.d1 = Dense(128, activation='relu')
    self.d2 = Dense(10)

  def call(self, x):
    x = self.flatten(x)
    x = self.d1(x)
    return self.d2(x)

# Create an instance of the model
model = MyModel()
print(model.inputs)
print(model.outputs)

from tensorflow import keras

class MyModel(keras.Model):
  def __init__(self):
    input = keras.Input(shape=(28, 28, 1), dtype="float32")
    x = keras.layers.Flatten()(input)
    x = keras.layers.Dense(128, activation='relu')(x)
    x = keras.layers.Dense(10)(x)
    super(MyModel, self).__init__(inputs=input, outputs=x)