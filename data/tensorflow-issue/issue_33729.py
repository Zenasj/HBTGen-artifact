# tf.random.uniform((B, 28, 28, 1), dtype=tf.float32) ← Input shape is MNIST images (batch size unspecified)

import tensorflow as tf
import tensorflow_probability as tfp

class MyModel(tf.keras.Model):
    def __init__(self, input_shape=(28, 28, 1), num_classes=10):
        super().__init__()
        # Bayesian convolutional layer with Flipout estimator, kernel_divergence_fn set to None per issue workaround
        # This avoids the "Graph tensor" error by disabling automatic KL divergence calculation,
        # which otherwise uses symbolic tensors incompatible with eager + tf.function.
        self.conv_flipout = tfp.layers.Convolution2DFlipout(
            6,
            kernel_size=5,
            padding="SAME",
            activation=tf.nn.relu,
            kernel_divergence_fn=None,
            input_shape=input_shape
        )
        self.flatten = tf.keras.layers.Flatten()
        self.dense_flipout_1 = tfp.layers.DenseFlipout(84, activation=tf.nn.relu)
        self.dense_flipout_2 = tfp.layers.DenseFlipout(num_classes)

    def call(self, inputs, training=False):
        x = tf.convert_to_tensor(inputs)
        x = self.conv_flipout(x, training=training)
        x = self.flatten(x)
        x = self.dense_flipout_1(x, training=training)
        x = self.dense_flipout_2(x, training=training)
        return x


def my_model_function():
    # Return an instance of MyModel with default MNIST input shape and 10 classes.
    # The model initializes layers and disables kernel_divergence_fn to avoid symbolic tensor issues.
    return MyModel()

def GetInput():
    # Returns a batch of one random MNIST-like input, normalized to [0,1].
    # Assumes data format 'channels_last' (28, 28, 1)
    # Batch size 1 chosen to keep runtime small.
    input_shape = (28, 28, 1)
    return tf.random.uniform(shape=(1, *input_shape), minval=0.0, maxval=1.0, dtype=tf.float32)

