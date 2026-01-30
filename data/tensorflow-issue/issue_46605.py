from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf

class MyModel(tf.keras.Model):

  def __init__(self, name='MyModel', **kwargs):
    super(MyModel, self).__init__(name=name, **kwargs)
    self.dense1 = tf.keras.layers.Dense(4, name='layer1')
    self.dense2 = tf.keras.layers.Dense(2, name='layer2')

  def call(self, inputs, training=None, mask=None):
    x = self.dense1(inputs)
    return self.dense2(x)

x_in = tf.keras.layers.Input(shape=(2,), dtype=tf.float32)
my_model = MyModel()
y = my_model(x_in, training=False)
mid_feat = my_model.get_layer('layer1').output