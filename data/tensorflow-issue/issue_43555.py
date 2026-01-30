from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
for ii in range(10):
  # Notice class definition *inside the loop*.
  class MyLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
      super(MyLayer, self).__init__(**kwargs)

    def call(self, inputs, training=None):
      return 2*inputs

  i1 = tf.keras.Input((1,))
  logits = MyLayer()(i1)
  model = tf.keras.Model(inputs=i1, outputs=logits)
  model.build(input_shape=(1,))
  model.predict(x=[[0.0], [1.0], [3.0]], batch_size=1)

import tensorflow as tf
for ii in range(10):
  # Notice class definition *inside the loop*.
  class MyLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
      super(MyLayer, self).__init__(**kwargs)

    def call(self, inputs, training=None):
      return inputs * 2

  b = tf.constant([5])
  i1 = tf.keras.Input(shape=(1,), tensor=b)
  logits = MyLayer()(i1)
  model = tf.keras.Model(inputs=i1, outputs=logits)
  #model.build(input_shape=(1,))
  print(model.predict(x=[[0.0], [1.0], [3.0]], batch_size=1))