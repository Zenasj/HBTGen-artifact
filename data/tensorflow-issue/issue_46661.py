# tf.random.uniform((B=1, H=256, W=256, C=3), dtype=tf.float64)

import tensorflow as tf
import numpy as np
import psutil
import tqdm


class LowLevelConvReLUModel:
    """
    A simple model implemented using low-level TF ops mimicking the problematic Conv2D+ReLU layers.
    This was used in issue to isolate the memory leak related to Conv2D + ReLU activation.
    """

    def __init__(self):
        self.built = False
        self.k1 = None
        self.b1 = None
        self.k2 = None
        self.b2 = None

    def __call__(self, inputs):
        if not self.built:
            # Note: Using float64 as in the original low-level test
            self.k1 = tf.Variable(tf.random.truncated_normal(
                (3, 3, 3, 64), 0.0, 0.1, dtype=tf.float64), trainable=True)
            self.b1 = tf.Variable(tf.zeros((64,), dtype=tf.float64), trainable=True)
            self.k2 = tf.Variable(tf.random.truncated_normal(
                (3, 3, 64, 64), 0.0, 0.1, dtype=tf.float64), trainable=True)
            self.b2 = tf.Variable(tf.zeros((64,), dtype=tf.float64), trainable=True)
            self.built = True

        y = inputs
        # First Conv2D + BiasAdd + ReLU
        y = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(
            y, self.k1, strides=[1, 1, 1, 1], padding='SAME'), self.b1))
        # Second Conv2D + BiasAdd + ReLU
        y = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(
            y, self.k2, strides=[1, 1, 1, 1], padding='SAME'), self.b2))
        return y


class KerasConvReLUModel(tf.keras.Model):
    """
    A Keras model composed of two Conv2D layers with ReLU activations,
    capturing the setup where the leak was observed with tf.function.
    """
    def __init__(self):
        super().__init__()
        self.layer_1 = tf.keras.layers.Conv2D(
            64, (3, 3), activation='relu', padding='same', name='block1_conv1')
        self.layer_2 = tf.keras.layers.Conv2D(
            64, (3, 3), activation='relu', padding='same', name='block1_conv2')

    def call(self, inputs):
        x = self.layer_1(inputs)
        x = self.layer_2(x)
        return x


class MyModel(tf.keras.Model):
    """
    Fused model that encapsulates both:
    - A Keras Conv2D+ReLU model
    - A low-level Conv2D + ReLU model implemented with raw ops and Variables
    The forward pass returns a dictionary of outputs, allowing comparison 
    as discussed in the issue thread.

    Wraps both models to allow comparison and debugging of memory leak issues.
    """

    def __init__(self):
        super().__init__()
        self.keras_model = KerasConvReLUModel()
        self.raw_model = LowLevelConvReLUModel()

    def call(self, inputs):
        """
        Inputs:
          - inputs : tf.Tensor of shape (1, 256, 256, 3), dtype=tf.float64.

        Outputs a dict with:
          - 'keras_out': output tensor from Keras model
          - 'raw_out': output tensor from low-level model
          - 'diff': element-wise absolute difference between outputs
          - 'close': tf.reduce_all of elementwise close within tolerance
        This format allows easy testing and comparison of outputs.
        """
        y_keras = self.keras_model(inputs)
        y_raw = self.raw_model(inputs)

        diff = tf.abs(y_keras - y_raw)
        # Use tf.experimental.numpy.allclose for numeric closeness test
        close = tf.experimental.numpy.allclose(y_keras, y_raw, rtol=1e-5, atol=1e-8)
        return {
            'keras_out': y_keras,
            'raw_out': y_raw,
            'diff': diff,
            'close': close
        }


def my_model_function():
    # Return an instance of MyModel
    return MyModel()


def GetInput():
    # Return a random tensor matching input shape (1,256,256,3),
    # dtype float64 matching examples in issue (low-level model used float64).
    return tf.random.uniform((1, 256, 256, 3), dtype=tf.float64)

