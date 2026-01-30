from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
from tensorflow.python.keras.testing_utils import layer_test

# a dummy layer that just returns a list of input
class MyCustomLayer(tf.keras.layers.Layer):
  def call(self, input):
    return [input, input]

layer_test(MyCustomLayer, input_shape=(1,2))