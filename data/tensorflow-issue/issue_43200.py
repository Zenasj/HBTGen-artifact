from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np

themodel = tf.keras.Sequential([
    tfa.layers.WeightNormalization(tf.keras.layers.Dense(1)) # breaks due to tf.bool [_initialzed](https://github.com/tensorflow/addons/blob/v0.11.2/tensorflow_addons/layers/wrappers.py#L105) layer weight
    #tf.keras.layers.Dense(1) # works fine
])

batch_size=1
x = np.zeros([batch_size, 10])
y = np.zeros([batch_size, 1])

themodel.compile(optimizer='sgd', loss='binary_crossentropy')
themodel.fit(x, y, batch_size=batch_size, epochs=1, callbacks=[tf.keras.callbacks.TensorBoard(log_dir='/tmp',histogram_freq=1,update_freq='epoch')])