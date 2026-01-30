from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf
import numpy as np
weights = np.ones((10, 11))
dense = tf.keras.layers.Dense(units=11, input_shape=(10,),
                          weights=[weights],
                          use_bias=False)
model = tf.keras.models.Sequential()
model.add(dense)
dense_init_weights = model.layers[0].get_weights()
np.allclose(weights,dense_init_weights)