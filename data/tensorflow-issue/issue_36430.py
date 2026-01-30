from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import torch
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import torch.nn as nn
import numpy as np
import os
import tensorflow as tf
import time

# tf.compat.v1.disable_eager_execution()

class TFReflectionPad1d(tf.keras.layers.Layer):
    def __init__(self, padding_size):
        super(TFReflectionPad1d, self).__init__()
        self.padding_size = padding_size
    
    def call(self, x):
        return tf.pad(x, [[0,0],[self.padding_size,self.padding_size],[0,0]], "REFLECT")
    

class TFUpsampleConv1d(tf.keras.layers.Layer):
    def __init__(self, upsample_factor, filters, kernel_size,
                 padding='same'):
        super(TFUpsampleConv1d, self).__init__()
        self.upsample1d = tf.keras.layers.UpSampling1D(size=upsample_factor)
        self.conv1d = tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, padding=padding)
    
    def call(self, x):
        x = self.upsample1d(x)
        return self.conv1d(x)
    
    
class TFResnetBlock(tf.keras.layers.Layer):
    def __init__(self, dim, dilation=1):
        super(TFResnetBlock, self).__init__()
        self.block = [
            tf.keras.layers.LeakyReLU(0.2),
            TFReflectionPad1d(dilation),
            tf.keras.layers.Conv1D(filters=dim, kernel_size=3, dilation_rate=dilation),
            tf.keras.layers.LeakyReLU(0.2),
            tf.keras.layers.Conv1D(filters=dim, kernel_size=1),
        ]
        self.shortcut = tf.keras.layers.Conv1D(filters=dim, kernel_size=1)
    
    def call(self, x):
        _x = tf.identity(x)
        for layer in self.block:
            _x = layer(_x)
        return self.shortcut(x) + _x

class TFMelGANGenerator(tf.keras.layers.Layer):
    def __init__(self, ngf, n_residual_layers):
        super(TFMelGANGenerator, self).__init__()
        ratios = [8,8,2,2]
        self.hop_length = np.prod(ratios)
        mult = int(2 ** len(ratios))
        
        model = [
            TFReflectionPad1d(3),
            tf.keras.layers.Conv1D(filters=mult * ngf, kernel_size=7, padding='valid')
        ]
        
        for i, r in enumerate(ratios):
            model += [
                tf.keras.layers.LeakyReLU(0.2),
                TFUpsampleConv1d(
                    upsample_factor=r,
                    filters=mult * ngf // 2,
                    kernel_size=r * 2 - 1,
                    padding='same'
                )
            ]
            
            for j in range(n_residual_layers):
                model += [TFResnetBlock(dim=mult * ngf // 2, dilation=3 ** j)]
            
            mult //= 2

        model += [
            tf.keras.layers.LeakyReLU(0.2),
            TFReflectionPad1d(3),
            tf.keras.layers.Conv1D(filters=1, kernel_size=7, padding='valid'),
            tf.keras.layers.Activation('tanh')
        ]
        self.model = tf.keras.models.Sequential(model)
    
    def call(self, x):
        return self.model(x)

inputs = tf.keras.Input(shape=[241, 80], dtype=tf.float32)
audio = TFMelGANGenerator(ngf=32, n_residual_layers=3)(inputs)
tf_melgan = tf.keras.models.Model(inputs, audio)

converter = tf.lite.TFLiteConverter.from_keras_model(tf_melgan)
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
converter.post_training_quantize = True
converter.experimental_new_converter = True
tflite_model = converter.convert()