# tf.random.uniform((B, 200, 6), dtype=tf.float32) ← Inferred input shape from issue description (batch size unknown, 200 time steps, 6 channels)
import tensorflow as tf
import numpy as np

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # The original model uses a TCN (Temporal Convolutional Network) from keras-tcn library,
        # but since we don't have the TCN code here, we create a placeholder network
        # mimicking input/output shapes and structure to allow conversion and quantization compatibility.
        # We use Conv1D and residual blocks to approximate a temporal CNN structure.

        self.conv1 = tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='causal', activation='relu')
        self.res_block1 = self._build_residual_block(64, 3)
        self.res_block2 = self._build_residual_block(64, 3)
        self.res_block3 = self._build_residual_block(64, 3)
        self.pool = tf.keras.layers.MaxPooling1D(pool_size=2)
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(50, activation='relu')
        self.dense_out = tf.keras.layers.Dense(2, activation='linear')  # output is 2 floating point scalars

    def _build_residual_block(self, filters, kernel_size):
        # Build a simple residual block with two Conv1D layers and skip connection
        return tf.keras.Sequential([
            tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, padding='causal', activation='relu'),
            tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, padding='causal', activation='linear')
        ])

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        # Residual block 1
        res = x
        out = self.res_block1(x)
        x = tf.nn.relu(out + res)
        # Residual block 2
        res = x
        out = self.res_block2(x)
        x = tf.nn.relu(out + res)
        # Residual block 3
        res = x
        out = self.res_block3(x)
        x = tf.nn.relu(out + res)

        x = self.pool(x)
        x = self.flatten(x)
        x = self.dense1(x)
        output = self.dense_out(x)  # shape (batch_size, 2)
        return output


def my_model_function():
    # Instantiate the model.
    model = MyModel()
    # Build the model by calling it on a sample input so that weights are created.
    dummy_input = tf.random.uniform((1,200,6), dtype=tf.float32)
    model(dummy_input)
    return model


def GetInput():
    # Return a random tensor input matching input expected by MyModel:
    # Shape: (batch_size, 200, 6), dtype=tf.float32
    # Since batch size is flexible, use batch size = 1 for simplicity
    return tf.random.uniform((1, 200, 6), dtype=tf.float32)

