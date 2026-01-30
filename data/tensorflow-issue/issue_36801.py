import random
from tensorflow import keras
from tensorflow.keras import layers

import json

import tensorflow as tf


def get_model(input_shape=(1,)):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=input_shape))
    model.add(tf.keras.layers.Dense(8, activation="relu"))
    model.add(tf.keras.layers.Dense(1))
    model.summary()
    model.compile(loss="mse", optimizer="adam")
    return model


def write_config(model, optimizer_file_path="optimizer.json"):
    with open(optimizer_file_path, "w") as f:
        json.dump(model.optimizer.get_config(), f, indent=4, sort_keys=True)


def get_data(n=100):
    import numpy as np
    data_x = np.random.rand(n, 1)
    data_y = np.asarray([1 if x > 0.5 else 0 for x in data_x]).reshape(data_x.shape)
    return data_x, data_y


if __name__ == '__main__':
    model = get_model()

    x, y = get_data()

    # No error before fitting.
    # write_config(model)

    model.fit(x, y, epochs=2)

    # Error after fitting.
    write_config(model)