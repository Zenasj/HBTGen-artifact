from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
import tensorflow.keras.layers as layers

class Variable(layers.Layer):

  def __init__(self, initial_value, **kwargs):
    super(Variable, self).__init__(**kwargs)
    self.var = tf.Variable(initial_value)

  def call(self, inputs):
    return self.var

var = Variable([1.0])([])
input = layers.Input(shape=(1,))
output = layers.Add()([input, var])
model = tf.keras.Model(inputs=[input], outputs=[output])
print(model(model.inputs))