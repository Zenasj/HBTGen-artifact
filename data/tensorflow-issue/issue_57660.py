import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

from typing import Tuple
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class ResizeLayer(tf.keras.layers.Layer):
    def resize_impl(self, elems):
        im, xy_x2y2 = elems
        x = xy_x2y2[0]
        y = xy_x2y2[1]
        width=xy_x2y2[2]-x
        height=xy_x2y2[3]-y
        # Ultimately Id like to run this too
        #im = tf.image.crop_to_bounding_box(im, y, x, height, width)
        im = tf.image.resize_with_pad(im, 90, 90)
        return im
    def call(self, x):
        im, coords = x
        coords = tf.cast(coords, tf.int32)
        return tf.map_fn(fn=self.resize_impl, elems=[im, coords],
            fn_output_signature=tf.TensorSpec((90,90,3), dtype=tf.float32))


i_im = tf.keras.layers.Input(shape=(None, None, 3))
i_c = tf.keras.layers.Input(shape=(4))
o = ResizeLayer()((i_im, i_c))

model = tf.keras.Model(inputs=(i_im, i_c), outputs=o)
model.summary()

im = tf.random.normal((1, 1200, 1200, 3), seed=42)
np.random.seed(42)
xy_x2y2 = tf.constant([[np.random.normal(loc=100), np.random.normal(loc=100), np.random.normal(loc=1000), np.random.normal(loc=1000)]])
gt = model((im, xy_x2y2))
model.save('mymodel')
model2 = tf.keras.models.load_model('mymodel')
got = model2((im, xy_x2y2))
print(tf.reduce_max(tf.abs(gt-got)))