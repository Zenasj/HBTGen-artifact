from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf

model = tf.keras.models.Sequential(
    layers=[tf.keras.layers.Dense(input_shape=(3, ), units=1)], 
    run_eagerly=True)