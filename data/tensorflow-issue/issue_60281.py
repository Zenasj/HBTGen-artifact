import random
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
import numpy as np
class EncoderBlock(tf.keras.layers.Layer):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.n_heads = 2
    def build(self, input_shape):
        self.WQ = self.add_weight("WQ_encoder", (self.n_heads, input_shape[-1], 10))
        # however, an easy fix is the using the following:
        # self.WQ = self.add_weight("WQ_encoder", (1, self.n_heads, input_shape[-1], 10))
        # but clearly there is a problem, as the broadcast should take place nicely like on Colab
    def call(self, inputs, mask, *args):
        inputs = inputs[:,None,...]
        _ = tf.matmul(inputs,self.WQ)
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^

EncoderBlock()(np.random.randn(3,10,13), np.ones((3,10)))