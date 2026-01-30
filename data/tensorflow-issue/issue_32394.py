import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

"""Example that reproduces the memory consumption increase during training."""
import numpy as np
import tensorflow as tf


def build_model():
    """Build a simple logistic regression model."""
    x = tf.keras.Input((100,))
    h = tf.keras.layers.Flatten()(x)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(h)
    return tf.keras.Model(inputs=[x], outputs=[output])


def load_data():
    """Load some dummy data."""
    x = np.random.randn(1000000, 100).astype(np.float32)
    y = np.random.choice([0, 1], size=(1000000, 1)).astype(np.float32)
    return x, y


def train(model, x, y):
    """Train the model on some data."""
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
                  loss=tf.keras.losses.binary_crossentropy)
    model.fit(x, y, epochs=10000, batch_size=1280)


if __name__ == '__main__':
    x, y = load_data()
    model = build_model()
    train(model, x, y)