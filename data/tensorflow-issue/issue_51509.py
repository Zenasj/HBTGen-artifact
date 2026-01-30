from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf

class MyModel(tf.keras.Model):

  def __init__(self, input_shape):
    super(MyModel, self).__init__()
    self.my_model_input_shape = input_shape
    self.dense1 = tf.keras.layers.Dense(5, activation=tf.nn.relu)
    self.dense2 = tf.keras.layers.Dense(5, activation=tf.nn.softmax)
    input_layer = tf.keras.layers.Input(self.my_model_input_shape)
    output_layer = self.call(input_layer)
    super(MyModel, self).__init__(
      inputs=input_layer,
      outputs=output_layer
    )

  def call(self, inputs, training=None):
    x = self.dense1(inputs)
    return self.dense2(x) + x


model_1 = MyModel((10,))
model_1.summary()

model_2 = MyModel((20,))
model_2.summary()