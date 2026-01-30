from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv1D(24, kernel_size = 2, dilation_rate = 2, padding='causal',
                           kernel_regularizer=tf.keras.regularizers.l2(0.01), input_shape=(2560, 8)),
    tf.keras.layers.ReLU(),
    tf.keras.layers.Dense(10),
    tf.keras.layers.Softmax(),
])

model.summary()

import tensorflow as tf
tf.enable_eager_execution()

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv1D(24, kernel_size = 2, dilation_rate = 2, padding='causal',
                           kernel_regularizer=tf.keras.regularizers.l2(0.01), input_shape=(2560, 8)),
    tf.keras.layers.ReLU(),
    tf.keras.layers.Dense(10),
    tf.keras.layers.Softmax(),
])

model.summary()