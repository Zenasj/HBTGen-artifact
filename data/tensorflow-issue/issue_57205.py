from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf


class Segment(tf.keras.layers.Layer):
    def __init__(self, ll, **kwargs):
        super().__init__(**kwargs)
        self._layers = tf.keras.Sequential()
        for l in (ll):
            if l == 'M':
                self._layers.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
            else:
                self._layers.add(tf.keras.layers.Conv2D(l, kernel_size=3, padding='same'))
                self._layers.add(tf.keras.layers.BatchNormalization())
                self._layers.add(tf.keras.layers.ReLU())
        self._layers.add(tf.keras.layers.AveragePooling2D(pool_size=1, strides=1))

    def call(self, x):
        return tf.recompute_grad(self._layers)(x)

import tensorflow as tf

from models.segment import Segment


config = {
    'vgg_new11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg_new13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg_new16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg_new19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}



class VGG(tf.keras.Model):
    def __init__(self, vgg_name, num_classes):
        super().__init__()
        model = config[vgg_name]
        self.blocks = tf.keras.Sequential()
        _seg = []
        for i, l in enumerate(model):
            _seg.append(l)
            if (i+1) % 3 == 0:
                self.blocks.add(Segment(_seg))
                _seg = []
            if (i+1) == len(model) and len(_seg) > 0:
                self.blocks.add(Segment(_seg))

        self.flatten = tf.keras.layers.Flatten()
        self.fc = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, x, **kwargs):
        out = self.blocks(x)
        out = self.flatten(out)
        out = self.fc(out)
        return out