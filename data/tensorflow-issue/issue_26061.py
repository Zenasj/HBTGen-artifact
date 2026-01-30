import random
from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf
from tensorflow import keras
import numpy as np

def my_glorot_initializer(shape, dtype=tf.float32):
    stddev = tf.sqrt(2. / (shape[0] + shape[1]))
    return tf.random.normal(shape, stddev=stddev, dtype=dtype)

def my_l1_regularizer(weights):
    return tf.reduce_sum(tf.abs(0.01 * weights))

def my_positive_weights(weights):
    return tf.nn.relu(weights)

X_train = np.random.randn(100, 2)
y_train = np.random.randn(100, 1)

model = keras.models.Sequential([
    keras.layers.Dense(1,
                       kernel_regularizer=my_l1_regularizer,
                       kernel_constraint=my_positive_weights,
                       kernel_initializer=my_glorot_initializer),
])

model.compile(loss="mse", optimizer="nadam")
model.fit(X_train, y_train, epochs=2)
model.save("my_model.h5")
model = keras.models.load_model(
    "my_model.h5",
    custom_objects={
       "my_l1_regularizer": my_l1_regularizer(0.01),
       "my_positive_weights": my_positive_weights,
       "my_glorot_initializer": my_glorot_initializer,
    })