# tf.random.uniform((B, 60), dtype=tf.float32) ← inferred input shape from code: input_shape=(60,)

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.utils import tf_utils

class GroupSoftmax(layers.Layer):
    def __init__(self, axis=-1, **kwargs):
        super(GroupSoftmax, self).__init__(**kwargs)
        self.supports_masking = True
        self.axis = axis

    def call(self, inputs):
        # The original code used tf.divide(inputs, tf.reduce_sum(inputs, axis=self.axis))
        # but that is not a stable softmax. Assuming intention was a group-wise softmax replacement,
        # Here we provide a sum normalization along axis.
        sum_along_axis = tf.reduce_sum(inputs, axis=self.axis, keepdims=True)
        # avoid division by zero
        sum_along_axis = tf.where(tf.equal(sum_along_axis, 0), tf.ones_like(sum_along_axis), sum_along_axis)
        return inputs / sum_along_axis

    def get_config(self):
        config = {'axis': self.axis}
        base_config = super(GroupSoftmax, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @tf_utils.shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape


class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        Nodes = 60  # inferred from usage of Dense layers with input_shape=(60,)
        self.seq = tf.keras.Sequential([
            layers.Dense(Nodes, activation='sigmoid', input_shape=(60,), use_bias=False),
            layers.Dense(Nodes, activation='sigmoid', use_bias=False),
            layers.Dense(Nodes, activation='sigmoid', use_bias=False),
            layers.Dense(Nodes, activation='sigmoid', use_bias=False),
            layers.Dense(Nodes, activation='sigmoid', use_bias=False),
            layers.Dense(66, activation='sigmoid', use_bias=False),
            layers.Reshape((11, 6)),
            GroupSoftmax(axis=0)  # Apply group softmax along axis 0 as original code
        ])

    @tf.function(jit_compile=True)
    def call(self, inputs):
        # Forward pass through the sequential model
        return self.seq(inputs)


def my_model_function():
    # Return an instance of MyModel
    return MyModel()


def GetInput():
    # Return a random tensor matching input expected by MyModel:
    # Input shape is (B, 60), batch size B can be arbitrary (e.g., 32)
    batch_size = 32
    # dtype inferred float32 (typical for keras input)
    return tf.random.uniform((batch_size, 60), dtype=tf.float32)

