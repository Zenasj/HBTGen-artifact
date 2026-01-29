# tf.random.normal((B, 8), dtype=tf.float32) ← Input shape inferred from the code: batch size B, feature size 8

import tensorflow as tf

def normalize_manual(x):
    mean = tf.reduce_mean(x, axis=-1, keepdims=True)
    x = x - mean
    var = tf.reduce_mean(tf.square(x), axis=-1, keepdims=True)
    return x / (tf.sqrt(var + 1e-6))


def normalize_with_moments(x):
    # Using tf.nn.moments (mean, variance) with keepdims=True
    mean, var = tf.nn.moments(x, axes=[-1], keepdims=True)
    x = x - mean
    return x / tf.sqrt(var + 1e-6)


class NormalizeManualModel(tf.keras.layers.Layer):
    def __init__(self, units=8):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(units)
        self.dense2 = tf.keras.layers.Dense(1)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = normalize_manual(x)
        x = self.dense2(x)
        return tf.squeeze(x, axis=-1)


class NormalizeMomentsModel(tf.keras.layers.Layer):
    def __init__(self, units=8):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(units)
        self.dense2 = tf.keras.layers.Dense(1)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = normalize_with_moments(x)
        x = self.dense2(x)
        return tf.squeeze(x, axis=-1)


class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Two sub-models differing by normalization method
        self.model_manual = NormalizeManualModel()
        self.model_moments = NormalizeMomentsModel()

    @tf.function(jit_compile=True)
    def call(self, inputs):
        # Forward pass through both models
        out_manual = self.model_manual(inputs)
        out_moments = self.model_moments(inputs)

        # Compare outputs with some tolerance
        # Return a boolean tensor indicating if close within tolerance
        close = tf.math.abs(out_manual - out_moments) < 1e-5
        return close, out_manual, out_moments


def my_model_function():
    # Return an instance of MyModel
    return MyModel()


def GetInput():
    # Generate a random input tensor of shape (batch_size=5, features=8)
    # Shape chosen arbitrarily to match example batches in the issue
    return tf.random.normal((5, 8))

