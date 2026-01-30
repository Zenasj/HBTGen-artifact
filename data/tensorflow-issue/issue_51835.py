from tensorflow.keras import layers
from tensorflow.keras import models

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Input
import tensorflow as tf
from model import load_gen

class ResAdd(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.res_gain = self.add_weight(shape=(), initializer='zeros', trainable=True)

    def call(self, inputs):
        res, skip = inputs
        gain = tf.cast(self.res_gain, res.dtype)
        out = res * gain + skip
        return out 

inp = Input((32, 32, 3)) 
out = ResAdd()([inp, inp])
model = Model(inp, out)

model.save('model')

self.res_gain = self.add_weight(shape=(), initializer='zeros', trainable=True)

self.res_gain = tf.Variable(0.0, trainable=True)

self.res_gain = self.add_weight(shape=(), initializer='zeros', trainable=True)

self.res_gain = tf.Variable(0.0, trainable=True)