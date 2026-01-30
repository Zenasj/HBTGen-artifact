from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

import numpy as np, tensorflow as tf

strategy = tf.distribute.MirroredStrategy()
path = "/tmp/model.hdf5"

with strategy.scope():
    # Construct model.
    model = tf.keras.models.Sequential([tf.keras.layers.Dense(1, input_shape=(1,))])
    model.compile(optimizer=tf.keras.optimizers.SGD(), loss=tf.keras.metrics.mse)
    # Do a fit so the optimizer weights are created. Removing this lets the restore succeed.
    model.fit(np.array([[1]]), np.array([[1]]))
    # Save and attempt to restore.
    tf.keras.models.save_model(model, path)
    tf.keras.models.load_model(path)

import tensorflow as tf

strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    model = tf.keras.models.Sequential([tf.keras.layers.Dense(1, input_shape=(1,))])
    model.compile(optimizer=tf.keras.optimizers.SGD(), loss=tf.keras.metrics.mse)
tf.keras.models.save_model(model, "/tmp/model", save_format="tf")

import numpy as np, tensorflow as tf

strategy = tf.distribute.MirroredStrategy()
path = "/tmp/model"

with strategy.scope():
    model = tf.keras.models.Sequential([tf.keras.layers.Dense(1, input_shape=(1,))])
    model.compile(optimizer=tf.keras.optimizers.SGD(), loss=tf.keras.metrics.mse)
    model.fit(np.array([[1]]), np.array([[1]]))
    tf.contrib.saved_model.save_keras_model(model, path)

    model = tf.contrib.saved_model.load_keras_model(path)
    model.compile(optimizer=tf.keras.optimizers.SGD(), loss=tf.keras.metrics.mse)
    model.fit(np.array([[1]]), np.array([[1]]))

class LoadWeightsCallback(tf.keras.callbacks.Callback):
    _chief_worker_only = False

    def __init__(self, weights, optimizer_weights):
        self.weights = weights
        self.optimizer_weights = optimizer_weights

    def on_train_begin(self, logs=None):
        self.model.set_weights(self.weights)
        self.model.optimizer.set_weights(self.optimizer_weights)