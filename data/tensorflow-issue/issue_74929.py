import random
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
import numpy as np

rng = np.random.default_rng()
data = rng.integers(-8, 8, [2, 4, 8, 2]).astype(np.uint16)

tf.keras.backend.clear_session()
x = tf.keras.Input(shape=[4, 8, 2], name='x', dtype=np.uint16)
y = tf.keras.layers.UpSampling2D(size=(2, 3), data_format='channels_last',
                                 interpolation='bilinear')(x)
model = tf.keras.Model(inputs=[x], outputs=[y])
res = model(data)
list(res)[0].dtype