from tensorflow import keras
from tensorflow.keras import layers

import numpy as np
import tensorflow as tf

def test_loss_function(loss_function):
    print("Testing {}".format(loss_function))

    layer = tf.keras.layers.Input(shape=(1,))
    model = tf.keras.Model(inputs=layer, outputs=layer)

    model.compile(optimizer='adam',
                  loss=loss_function)

    weights = np.array([1., 1., 1., 1., 1., 0., 0., 0., 0., 0.])
    model.evaluate(np.zeros(10), np.ones(10), sample_weight=weights)

test_loss_function('mean_squared_error')
test_loss_function(tf.keras.losses.mean_squared_error)