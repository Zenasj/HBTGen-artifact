import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf


class Bad(tf.keras.models.Model):
    def __init__(self):
        super(Bad, self).__init__()
        # self.rec_net = tf.keras.layers.LSTM(10)
        self.rec_net = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(10))
    
    @tf.function(input_signature=
                 [tf.TensorSpec(shape=(None, None, 1), dtype=tf.float32),
                  tf.TensorSpec(shape=(None, None, 1), dtype=tf.int32), ])
    def call(self, x, mask):
        return self.rec_net(x, mask=tf.cast(mask, tf.bool), training=False)
        # return self.rec_net(x, training=False)


if __name__ == '__main__':
    inp = tf.random.uniform((3, 4, 1))
    mask = tf.convert_to_tensor([[[1], [1], [1], [1]],
                                 [[1], [1], [1], [0]],
                                 [[1], [1], [0], [0]]])
    
    b = Bad()
    out = b(inp, mask)