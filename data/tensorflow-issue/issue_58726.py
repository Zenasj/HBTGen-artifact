import random

import numpy as np
import tensorflow as tf
from keras import layers as tfl

class Encoder(tfl.Layer):
    def __init__(self,):
        super().__init__()
        self.embed_layer = tfl.Embedding(4500, 32, mask_zero=True)
        self.attn_layer = tfl.MultiHeadAttention(num_heads=2,
                                                 attention_axes=2,
                                                 key_dim=16)
        return

    def call(self, x):
        # Input shape: (256, 10, 20) (Batch size: 256)
        x = self.embed_layer(x)  # Output: (256, 10, 20, 32)
        x = self.attn_layer(query=x, key=x, value=x)  # Output: (256, 10, 20, 32)
        return x


eg_input = tf.constant(np.random.randint(0, 150, (256, 12, 20)))
enc = Encoder()
enc(eg_input).shape

def call(self, x):
        # Input shape: (256, 10, 20) (Batch size: 256)
        x = self.embed_layer(x)  # Output: (256, 10, 20, 32)
        x = tf.concat(x, axis=0) # Output: (256, 10, 20, 32)
        x = self.attn_layer(query=x, key=x, value=x)  # Output: (256, 10, 20, 32)
        return x