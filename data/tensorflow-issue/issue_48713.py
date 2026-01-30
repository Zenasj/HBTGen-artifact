from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
import numpy as np
import random


def test_optimizer(optimizer):
  class Layer1(tf.keras.layers.Layer):

      def __init__(self, shape):
          super(Layer1, self).__init__()
          self.weight1 = self.add_weight(shape=shape)
          
      def call(self, inputs):
          return tf.gather(self.weight1, axis=0, indices=inputs)


  x = tf.keras.Input(shape=[1,])
  ind = tf.cast(x, "int64")
  y = Layer1(shape=(10, 2))(ind)
  model = tf.keras.Model(x, y)
  model.compile(optimizer=optimizer, loss='mse')

  x_data = np.array([[random.randint(0, 9),],])

  out = model(x_data)
  y_data = np.zeros(out.shape)
  with tf.GradientTape() as tape:
    y_pred = model(x_data)
    loss = tf.keras.losses.MeanSquaredError()(y_data, y_pred)

    gradients = tape.gradient(loss, model.trainable_weights)

  optimizer.apply_gradients(zip(gradients, model.trainable_weights))

test_optimizer(tf.optimizers.Adam())
print('Adam Passed')
test_optimizer(tf.optimizers.Adadelta())