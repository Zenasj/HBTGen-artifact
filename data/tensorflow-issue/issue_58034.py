import random
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf


class Model(tf.keras.Model):

  def __init__(self):
    super().__init__()
    self.blocks = [
        tf.keras.layers.Conv2D(filters=32,
                               kernel_size=(3, 3),
                               strides=(1, 1),
                               padding='same',
                               data_format='channels_first',
                               name='upscale')
    ]

  def call(self, inputs):
    layer = inputs
    for b in self.blocks:
      layer = b(layer)
    return layer


def main():
  m = Model()
  x = [tf.random.uniform(shape=(32,90,180)) for _ in range(32)]
  y = [tf.random.uniform(shape=(32,90,180)) for _ in range(32)]
  ds = tf.data.Dataset.from_tensor_slices((x,y)).batch(16)
  m.compile(jit_compile=True, loss='mse')
  m.fit(ds, epochs=5)

if __name__ == '__main__':
  main()

import tensorflow as tf
sys_details = tf.sysconfig.get_build_info()