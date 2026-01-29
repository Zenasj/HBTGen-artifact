# tf.random.uniform((10, 128, 128, 9), dtype=tf.float32)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Weight initializer similar to original example
        self.weights_initializer = tf.compat.v1.initializers.truncated_normal(
            mean=0.0,
            stddev=tf.math.sqrt(2.0 / ((3 ** 2) * 32))
        )
        self.activation_fn = tf.keras.layers.LeakyReLU()
        # First conv2d: input channels=9, output channels=32
        self.conv1 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            strides=1,
            padding='same',
            kernel_initializer=self.weights_initializer,
            kernel_regularizer=tf.keras.regularizers.l2(1.0),
            bias_regularizer=tf.keras.regularizers.l2(1.0),
            activation=self.activation_fn
        )
        # Second conv2d: input channels=32, output channels=3
        self.conv2 = tf.keras.layers.Conv2D(
            filters=3,
            kernel_size=(3, 3),
            strides=1,
            padding='same',
            kernel_initializer=self.weights_initializer,
            kernel_regularizer=tf.keras.regularizers.l2(1.0),
            bias_regularizer=tf.keras.regularizers.l2(1.0),
            activation=self.activation_fn
        )

    def call(self, inputs, training=False):
        # Forward pass through conv layers
        x = self.conv1(inputs)
        x = self.conv2(x)
        return x

def my_model_function():
    # Instantiate and return MyModel
    return MyModel()

def GetInput():
    # Return random input tensor matching expected input shape of MyModel
    # Shape: batch=10, height=128, width=128, channels=9 (as in the issue example)
    return tf.random.uniform((10, 128, 128, 9), dtype=tf.float32)

