import random
from tensorflow import keras
from tensorflow.keras import layers

import numpy as np
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(20,)),
    tf.keras.layers.Dense(5)
])

# With the following line uncommented, it works.
# model(model.input)

def loss(y_true, y_pred):
    loss = tf.keras.losses.mean_squared_error(y_true, y_pred)

    # Feed zeros to the model and add the mean of the output to the loss
    y_pred_zeros = model(tf.zeros((32, 20)))

    return loss + tf.reduce_mean(y_pred_zeros)

model.compile('sgd', loss=loss)

x_train, y_train = np.random.randn(100, 20), np.random.randn(100, 5)
model.fit(x_train, y_train)