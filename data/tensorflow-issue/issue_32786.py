from tensorflow import keras
from tensorflow.keras import layers

python
import tensorflow as tf

shape = (1, 2)


class MyLayer(tf.keras.layers.Layer):
    def build(self, input_shape):
        print(input_shape)
        assert input_shape == shape

    def call(self, inputs):
        return inputs


layer = MyLayer()
layer.compute_output_shape(shape)

import tensorflow as tf

shape = (1, 2)


class MyLayer(tf.keras.layers.Layer):

  def __init__(self, **kwargs):
        super(MyLayer, self).__init__(**kwargs)
        
  def build(self, input_shape):
      print(input_shape)
      super(MyLayer, self).build(input_shape)
      assert input_shape == shape

  def call(self, inputs):
      return inputs


layer = MyLayer()
layer.build(shape)
layer.compute_output_shape(shape)