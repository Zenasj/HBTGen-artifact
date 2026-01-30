import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import numpy as np
import tensorflow as tf


class CustomLayer(tf.keras.layers.Layer):
    """Imaginary Layer that adds a custom loss in the call"""

    def __init__(self, a):
        super().__init__()
        self.var = tf.Variable(a, name='var_a')

    def call(self, inputs, training=False):
        output = tf.reduce_sum(inputs * self.var, axis=-1)
        self.add_loss(tf.reduce_mean(output))
        return output


def get_model(input_dim: int) -> tf.keras.Model:
    layer = CustomLayer(0.1)
    inputs = tf.keras.Input((input_dim,), name="inputs")
    outputs = layer(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam')
    return model


if __name__ == "__main__":
    num_data = 100
    X, Y = np.random.randn(num_data, 1), np.random.randn(num_data, 1)

    model = get_model(input_dim=X.shape[-1])
    model.summary()
    print("model.losses", model.losses)

    model.save('my_model')
    reconstructed_model = tf.keras.models.load_model('my_model')  # ~~ breaks 
    reconstructed_model.summary()

    # Let's check:
    np.testing.assert_allclose(
        model.predict(X),
        reconstructed_model.predict(X)
    )

reconstructed_model = tf.keras.models.load_model('my_model', custom_objects={"CustomLayer": CustomLayer})

print("model.losses:", model.losses)
print("reconstructed_model.losses:", reconstructed_model.losses)