import math
import random
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf

from keras import backend as K
from tensorflow.keras.constraints import Constraint
from tensorflow.python.util.tf_export import keras_export
from tensorflow.python.keras.utils import conv_utils
# import skimage.measure

@keras_export('keras.constraints.UnitSumNonNeg', 'keras.constraints.unit_sum_non_neg')
class UnitSumNonNeg(Constraint):
    """Limits weights to be non-negative and with sum equal to one

    Also available via the shortcut function `keras.constraints.unit_sum_non_neg`.
    """
    def __call__(self, w):
        aux =  w * tf.cast(tf.math.greater_equal(w, 0.), w.dtype)

        return aux/(K.epsilon() + tf.reduce_sum(aux, axis=[0], keepdims=True))

class OWAPoolingNew(tf.keras.layers.Layer):
    def __init__(self,
               pool_size=(2, 2),
               strides=None,
               padding='valid',
               data_format=None,
               name=None,
               sort=True,
               train=True, 
               seed=None,
               all_channels=False,
               **kwargs):
        super(OWAPoolingNew, self).__init__(name=name, **kwargs)

        self.pool_size = pool_size
        self.strides = pool_size if strides == None else strides
        self.padding = padding
        self.data_format = conv_utils.normalize_data_format('channels_last')
        self.sort = sort
        self.train = train
        self.seed = seed if seed != None else 10
        self.all_channels = all_channels
        
    def build(self, input_shape):
      
      if self.all_channels:
        weights_shape = (self.pool_size[0] * self.pool_size[1], input.shape[-1])
      else:
        weights_shape = (self.pool_size[0] * self.pool_size[1], 1)
      
      tf.random.set_seed(self.seed)
      kernel = tf.random.uniform(shape=weights_shape)
      kernel /= tf.reduce_sum(kernel, axis=[0], keepdims=True)
      
      self.kernel = tf.Variable(initial_value = kernel, trainable=self.train, dtype='float32', constraint=UnitSumNonNeg())

    def call(self, inputs):

        _, height, width, channels = inputs.get_shape().as_list()

        # Extract pooling regions
        stride = [1, self.strides[0], self.strides[1], 1]
        ksize = [1, self.pool_size[0], self.pool_size[1], 1]

        inputs = tf.image.extract_patches(inputs, sizes = ksize, strides = stride,
                            rates = [1, 1, 1, 1], padding='SAME')

        _, pool_height, pool_width, elems = inputs.get_shape().as_list()

        # Extract pooling regions for each channel
        elems =  int(elems / channels)
        inputs = tf.reshape(inputs, [-1, pool_height, pool_width, elems, channels]) # Reshape tensor

        # Sort values for pooling
        if self.sort:
            inputs = tf.sort(inputs, axis=-2, direction='DESCENDING', name=None)

        outputs = tf.reduce_sum(tf.math.multiply(self.kernel, inputs), axis=-2)

        return outputs