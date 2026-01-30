from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import os
import tempfile

import tensorflow as tf
import numpy as np


def test_bug_tensorflow():

    nb_inputs = 784
    hidden_units = [32, 32]
    nb_classes = 10

    # backbone model
    input_1 = tf.keras.layers.Input(shape=(nb_inputs,), name="input_backbone", dtype=tf.float32)
    x = input_1
    for i, units in enumerate(hidden_units):
        x = tf.keras.layers.Dense(units, name=f"dense_{i}", activation="relu")(x)
    backbone = tf.keras.models.Model(input_1, x, name="backbone")

    # classifier
    input_2 = tf.keras.layers.Input(shape=(nb_inputs,), name="input_mnist", dtype=tf.float32)
    x = backbone(input_2)
    x = tf.keras.layers.Dense(nb_classes, activation="softmax", name="output")(x)
    mnist_model = tf.keras.models.Model(input_2, x, name="mnist")
    mnist_model.predict(np.zeros((100, nb_inputs)))
    print(mnist_model.summary())

    tmp_dir = tempfile.TemporaryDirectory()
    tmp_path = str(tmp_dir.name)
    
    mnist_model.save(os.path.join(tmp_path, "model.h5"), save_format="h5")
    new_model = tf.keras.models.load_model(os.path.join(tmp_path, "model.h5"))
    mnist_model.predict(np.zeros((100, nb_inputs)))
    print(new_model.summary())

    mnist_model.save(tmp_path, save_format="tf")
    new_model = tf.keras.models.load_model(tmp_path)
    mnist_model.predict(np.zeros((100, nb_inputs)))
    print(new_model.summary())  # raises ValueError