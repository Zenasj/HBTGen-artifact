from tensorflow import keras
from tensorflow.keras import layers

import numpy as np
import tensorflow as tf

m = tf.keras.Sequential([
    tf.keras.layers.Masking(.0),
    tf.keras.layers.GlobalMaxPool1D(),
])
print(m(np.array([[[0.]]], np.float32))._keras_mask)