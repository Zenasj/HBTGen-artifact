import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

from tensorflow.python import pywrap_tensorflow

model = tf.keras.models.Sequential([
  tf.keras.layers.Input(shape=(mat_size_aug,mat_size_red,1)),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(units = out_size,  name="dense_out",
    activation = 'linear', use_bias=False)
])