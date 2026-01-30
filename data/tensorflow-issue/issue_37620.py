from tensorflow import keras
from tensorflow.keras import layers

#!/usr/bin/env python3

import tensorflow as tf
import numpy as np


class TestModel(tf.keras.Model):
    def __init__(self, n_layers, **kwargs):
        super(TestModel, self).__init__(**kwargs)

        self.n_layers = n_layers
        self.map = {}
        for i in range(self.n_layers):
            self.map[str(i)] = tf.keras.layers.Dense(units=1)

    def call(self, X, training=False):

        for i in range(self.n_layers):
            X = self.map[str(i)](X)

        return X


if __name__ == "__main__":
    n_layers = 5

    # Custom model
    custom = TestModel(n_layers)
    custom.compile(optimizer="adam", loss="mse")
    custom.fit(
        x=np.ones((10000), np.float32),
        y=np.ones((10000), np.float32),
        epochs=10,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                filepath="output/custom/callback/model{epoch}", save_weights_only=False,
            )
        ],
    )
    custom.save("output/custom/manual/model")

    # Sequential model
    sequential = tf.keras.Sequential([tf.keras.layers.Dense(units=1) for i in range(n_layers)])
    sequential.compile(optimizer="adam", loss="mse")
    sequential.fit(
        x=np.ones((10000), np.float32),
        y=np.ones((10000), np.float32),
        epochs=10,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                filepath="output/sequential/callback/model{epoch}",
                save_weights_only=False,
            )
        ],
    )
    sequential.save("output/sequential/manual/model")