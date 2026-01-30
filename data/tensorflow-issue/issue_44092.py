from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf

class DenseBlock(tf.keras.Model):
  def __init__(self, input_size, depth=5, in_channels=64):
    super(DenseBlock, self).__init__(name='')
    self.depth = depth
    self.in_channels = in_channels
    self.pad = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 1)))
    self.twidth = 2
    self.kernel_size = (self.twidth, 3)
    for i in range(self.depth):
      dil = 2**i
      pad_length = self.twidth + (dil-1)*(self.twidth-1)-1
      setattr(self, 'pad{}'.format(i+1), tf.keras.layers.ZeroPadding2D(((pad_length, 0), (1, 1))))
      setattr(self, 'conv{}'.format(i+1), tf.keras.layers.Conv2D(filters=self.in_channels, kernel_size=self.kernel_size, dilation_rate=(dil, 1)))
      setattr(self, 'norm{}'.format(i+1), tf.keras.layers.LayerNormalization())
      setattr(self, 'prelu{}'.format(i+1), tf.keras.layers.PReLU(shared_axes=[1, 2]))

  def call(self, input_tensor):
    skip = input_tensor
    for i in range(self.depth):
      print('Dilation rate', 2**i)
      x = getattr(self, 'pad{}'.format(i+1))(skip)
      print(x.shape)
      x = getattr(self, 'conv{}'.format(i+1))(x)
      print(x.shape)
      x = getattr(self, 'norm{}'.format(i+1))(x)
      x = getattr(self, 'prelu{}'.format(i+1))(x)
      skip = tf.concat((x, skip), axis=3)
    return x

input = tf.keras.layers.Input(shape=(None, 512, 64))
x = DenseBlock(512, 5, 64)(input)