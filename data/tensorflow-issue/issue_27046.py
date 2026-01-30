import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class DenseModule(tf.Module):
  def __init__(self):
    super(DenseModule, self).__init__()
    self._layer = tf.keras.layers.Dense(64)

  def __call__(self, x):
    y = self._layer(x)
    # Keep a reference to layer variables so Module can find them.
    self._k_variables = self._layer.variables
    return y