import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf
import numpy as np


def replace_conv2d_with_sub_model(layer: tf.keras.layers.Layer) -> tf.keras.layers.Layer:
    if isinstance(layer, tf.keras.layers.Conv2D):
        inputs = layer.input
        x = layer(inputs)
        y = tf.keras.layers.Lambda(lambda t: t * 2)(x)
        outputs = tf.keras.layers.Add()([x, y])
        layer = tf.keras.models.Model(inputs=inputs, outputs=outputs, name=layer.name)
    return layer


def get_sequential_model() -> tf.keras.Model:
    return tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(32, 3, activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(10, activation='softmax'),
        ]
    )


def get_functional_model() -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(28, 28, 1))
    x = tf.keras.layers.Conv2D(32, 3, activation='relu')(inputs)
    x = tf.keras.layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(10, activation='softmax')(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)


if __name__ == '__main__':
    model = get_functional_model()  # get_sequential_model() works
    model.compile()
    sample_input = np.random.uniform(size=(1, 28, 28, 1))
    o1 = model.predict(sample_input)

    new_model = tf.keras.models.clone_model(
        model, input_tensors=model.inputs, clone_function=replace_conv2d_with_sub_model
    )  # <-- results in an infinite loop

    o2 = new_model.predict(sample_input)