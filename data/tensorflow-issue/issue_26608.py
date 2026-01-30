import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import tensorflow as tf


class DebugModel(tf.keras.Model):
  def __init__(self):
    super(DebugModel, self).__init__()
    self.dense = tf.keras.layers.Dense(20, activation='relu', input_shape=(5,))

  def __call__(self, input, *args, **kwargs):
    x = self.dense(input)
    ind = [[0], [1], [2]]
    x = tf.gather_nd(x, ind)
    ind = [[0], [1]]
    x = tf.gather_nd(x, ind)
    return x

my_model = DebugModel()

my_optim = tf.keras.optimizers.Adam(1e-3)
with tf.GradientTape() as tape:
  out = my_model(tf.random.uniform((16,5)))
  my_loss = -tf.reduce_mean(out)
  grads = tape.gradient(my_loss, my_model.trainable_weights)
  my_optim.apply_gradients(zip(grads, my_model.trainable_weights))