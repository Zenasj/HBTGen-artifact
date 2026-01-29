# tf.random.uniform((B, 96, 96, 3), dtype=tf.float16)

import tensorflow as tf
import numpy as np

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()

        # Conv2D layer with Orthogonal initializer using gain sqrt(2)
        self.conv1 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=(8, 8),
            strides=(4, 4),
            padding="same",
            kernel_initializer=tf.keras.initializers.Orthogonal(np.sqrt(2)),
            bias_initializer=tf.keras.initializers.Zeros(),
            activation="linear"  # Activation applied separately below (LeakyReLU)
        )

        self.leaky_relu = tf.keras.layers.LeakyReLU()

        self.flatten = tf.keras.layers.Flatten()

        # Dense layer with Orthogonal initializer using gain sqrt(2)
        self.a_dense = tf.keras.layers.Dense(
            512,
            kernel_initializer=tf.keras.initializers.Orthogonal(np.sqrt(2)),
            bias_initializer=tf.keras.initializers.Zeros(),
            activation="relu"
        )

        # Output dense layer with 9 units and softmax activation
        self.a_out = tf.keras.layers.Dense(
            9,
            bias_initializer=tf.keras.initializers.Zeros(),
            activation="softmax"
        )

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.leaky_relu(x)
        x = self.flatten(x)
        x = self.a_dense(x)
        x = self.a_out(x)
        return x

def my_model_function():
    # Set backend floatx to float16 as per issue context
    # In practice, changing backend floatx in model factory is usually avoided here,
    # but it's included for faithful reproduction of the issue's environment.
    tf.keras.backend.set_floatx("float16")
    return MyModel()

def GetInput():
    # Input tensor of shape (batch=1, height=96, width=96, channels=3), dtype float16
    # Using uniform random values in [0,1) as a general input
    return tf.random.uniform(shape=(1, 96, 96, 3), dtype=tf.float16)

