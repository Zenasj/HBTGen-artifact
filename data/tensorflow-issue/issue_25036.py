# tf.random.uniform((B, 28, 28, 1), dtype=tf.float32)
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model, Sequential


class MyModel(tf.keras.Model):
    """
    This MyModel fuses two main subclassed CNN models presented in the issue:
    - ClassCNN: a typical ConvNet with explicit .build() calls for shape inference
    - ClassNN: a fully connected model flattening the input with hidden dense layers and dropout

    It implements a simple comparison between outputs of these two models for the same input,
    returning a boolean tensor indicating if the outputs are "close" within a tolerance.

    This satisfies the requirement to fuse multiple models discussed jointly into one model,
    exposing their comparison output on forward pass.
    """

    def __init__(self, input_shape=(28, 28, 1), output_shape=10, atol=1e-5, rtol=1e-3):
        super().__init__()

        self._input_shape = input_shape
        self._output_shape = output_shape
        self.atol = atol
        self.rtol = rtol

        # ClassCNN part (ConvNet)
        self.conv1 = Conv2D(32, kernel_size=(3, 3), activation=tf.nn.relu)
        # After conv1: shape (None, 26, 26, 32)
        self.conv2 = Conv2D(64, kernel_size=(3, 3), activation=tf.nn.relu)
        # After conv2: shape (None, 24, 24, 64)
        self.maxpool = MaxPooling2D(pool_size=(2, 2))
        # After maxpool: shape (None, 12, 12, 64)
        self.dropout1 = Dropout(0.25, name='dropout1')
        self.flatten = Flatten()
        self.dense1 = Dense(128, activation=tf.nn.relu)
        self.dropout2 = Dropout(0.5, name='dropout2')
        self.activation = Dense(self._output_shape, activation=tf.nn.softmax)

        # Explicitly build layers to fix output shapes for subclassed model summary
        self.conv1.build((None,) + input_shape)
        # input_shape[:-1] = (28, 28), subtract 2 for kernel effect
        conv1_out_shape = (input_shape[0] - 2, input_shape[1] - 2, 32)
        self.conv2.build((None,) + conv1_out_shape)
        conv2_out_shape = (conv1_out_shape[0] - 2, conv1_out_shape[1] - 2, 64)
        self.maxpool.build((None,) + conv2_out_shape)
        # maxpool halves width and height by floor div
        maxpool_out_shape = (conv2_out_shape[0] // 2, conv2_out_shape[1] // 2, 64)
        self.dropout1.build((None,) + maxpool_out_shape)
        # flatten output dims = product of spatial dims * channels
        flattened_dim = maxpool_out_shape[0] * maxpool_out_shape[1] * 64
        self.dropout2.build((None, 128))
        self.dense1.build((None, flattened_dim))
        self.flatten.build((None,) + maxpool_out_shape)
        self.activation.build((None, 128))
        self.build((None,) + input_shape)

        # ClassNN part (Fully connected network)
        self.flatten_nn = Flatten()
        self.dense1_nn = Dense(128, activation=tf.nn.relu)
        self.dropout_nn = Dropout(0.2)
        self.dense2_nn = Dense(128, activation=tf.nn.softmax)
        self.activation_nn = Dense(self._output_shape, activation=tf.nn.softmax)

        input_nn_dim = np.prod(input_shape)
        self.flatten_nn.build((None,) + input_shape)
        self.dense1_nn.build((None, input_nn_dim))
        self.dropout_nn.build((None, 128))
        self.dense2_nn.build((None, 128))
        self.activation_nn.build((None, 128))

        # No separate build on this model, subclass will have summary()

    def call(self, inputs, training=False, **kwargs):
        # Forward pass through ClassCNN submodel
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.maxpool(x)
        if training:
            x = self.dropout1(x)
        x = self.flatten(x)
        x = self.dense1(x)
        if training:
            x = self.dropout2(x)
        output_cnn = self.activation(x)

        # Forward pass through ClassNN submodel
        y = self.flatten_nn(inputs)
        y = self.dense1_nn(y)
        if training:
            y = self.dropout_nn(y)
        y = self.dense2_nn(y)
        output_nn = self.activation_nn(y)

        # Compare outputs (elementwise closeness) with given tolerances
        close = tf.math.abs(output_cnn - output_nn) <= (self.atol + self.rtol * tf.math.abs(output_nn))

        # Return boolean tensor reflecting elementwise closeness as final output
        return close


def my_model_function():
    # Return an instance of MyModel with default input/output shapes (28x28 grayscale, 10 classes)
    return MyModel(input_shape=(28, 28, 1), output_shape=10)


def GetInput():
    # Generate a random input tensor matching expected input for MyModel
    # Batch size is chosen as 1 for simplicity; dtype float32 matches typical image data
    return tf.random.uniform(shape=(1, 28, 28, 1), dtype=tf.float32)

