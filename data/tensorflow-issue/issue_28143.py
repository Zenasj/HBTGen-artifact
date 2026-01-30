from tensorflow import keras
from tensorflow.keras import layers

import numpy as np
import tensorflow as tf
m = tf.keras.Sequential([
    tf.keras.layers.Masking(.0),
    tf.keras.layers.Lambda(tf.nn.sigmoid)])
#     tf.keras.layers.GlobalMaxPool1D()])
m(np.array([[[0.]]]))