# tf.random.uniform((B, input_dim), dtype=tf.float32)  ← Assuming input shape is (batch_size, input_dim)

import tensorflow as tf
from tensorflow.keras import backend as K

class RBFLayer(tf.keras.layers.Layer):
    def __init__(self, units, gamma, **kwargs):
        super(RBFLayer, self).__init__(**kwargs)
        self.units = units
        self.gamma = tf.cast(gamma, dtype=tf.float32)

    def build(self, input_shape):
        # Expect input_shape = (batch_size, input_dim)
        input_dim = int(input_shape[1])
        # Centers (mu) shape: (input_dim, units)
        # This is somewhat unusual, often mu is (units, input_dim) for RBF centers.
        # Here we keep the original shape but transpose in call if needed.
        self.mu = self.add_weight(
            name='mu',
            shape=(input_dim, self.units),
            initializer=tf.random_normal_initializer(),
            trainable=True,
            dtype=tf.float32)
        super(RBFLayer, self).build(input_shape)

    def call(self, inputs):
        # inputs: (batch_size, input_dim)
        # mu: (input_dim, units)
        # Expand inputs to (batch_size, 1, input_dim) and mu to (1, input_dim, units)
        # But with shape (input_dim, units), we must adjust to compute difference correctly.

        # To compute difference: expand inputs to (batch_size, input_dim, 1)
        inputs_expanded = tf.expand_dims(inputs, axis=2)  # (B, input_dim, 1)
        # mu is (input_dim, units)
        mu_broadcast = tf.expand_dims(self.mu, axis=0)  # (1, input_dim, units)

        # difference: (B, input_dim, units)
        diff = inputs_expanded - mu_broadcast

        # squared L2 distance across input_dim axis (axis=1)
        l2 = tf.reduce_sum(tf.square(diff), axis=1)  # (B, units)

        # Apply RBF kernel
        res = tf.exp(-0.5 * self.gamma * l2)  # (B, units)
        return res

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)

class MyModel(tf.keras.Model):
    def __init__(self, units=10, gamma=1.0):
        super(MyModel, self).__init__()
        # We create an RBF layer instance as a submodule
        self.rbf = RBFLayer(units=units, gamma=gamma)

    def call(self, inputs):
        return self.rbf(inputs)

def my_model_function():
    # Create MyModel with some default parameters
    # Units and gamma can be changed as needed
    return MyModel(units=10, gamma=1.0)

def GetInput():
    # Return a random tensor input that matches expected input shape
    # Based on RBFLayer build method, input shape is (batch_size, input_dim)
    # input_dim was inferred from shape of mu (input_dim, units)
    # We'll pick input_dim=5 arbitrarily as common dimension, batch=8

    batch_size = 8
    input_dim = 5
    # Use float32 as K.cast_to_floatx uses float32 as default
    return tf.random.uniform(shape=(batch_size, input_dim), dtype=tf.float32)

