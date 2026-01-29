# tf.ragged.constant with shape (batch_size, None, 1), dtype=float32

import tensorflow as tf
import numpy as np

class DenseRagged(tf.keras.layers.Layer):
    def __init__(self, 
                 units,
                 use_bias=True,
                 activation='linear',
                 **kwargs):
        super(DenseRagged, self).__init__(**kwargs)
        # This layer supports ragged inputs
        self._supports_ragged_inputs = True 
        self.units = units
        self.use_bias = use_bias
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        # Last dimension of input shape corresponds to feature dimension
        last_dim = input_shape[-1]
        self.kernel = self.add_weight(
            name='kernel',
            shape=[last_dim, self.units],
            trainable=True)
        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=[self.units,],
                trainable=True)
        else:
            self.bias = None
        super(DenseRagged, self).build(input_shape)

    def call(self, inputs):
        # Map dense matmul, bias add, and activation over ragged flat values
        outputs = tf.ragged.map_flat_values(tf.matmul, inputs, self.kernel)
        if self.use_bias:
            outputs = tf.ragged.map_flat_values(tf.nn.bias_add, outputs, self.bias)
        outputs =  tf.ragged.map_flat_values(self.activation, outputs)
        return outputs


class PoolingRagged(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(PoolingRagged, self).__init__(**kwargs)
        self._supports_ragged_inputs = True

    def build(self, input_shape):
        super(PoolingRagged, self).build(input_shape)

    def call(self, inputs):
        # inputs shape: (batch_size, None, feature_dim), ragged at axis=1
        # Reduce mean along axis 1 (ragged dimension)
        # tf.reduce_mean works directly on ragged by computing mean per sublist
        out = tf.math.reduce_mean(inputs, axis=1)
        return out


class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # The model structure fuses the two DenseRagged+PoolingRagged paths in one,
        # analogous to model2 in the issue example.
        self.dense_ragged = DenseRagged(1)
        self.pooling_ragged = PoolingRagged()

    def call(self, inputs):
        # inputs is expected to be a list/tuple of two ragged tensors: [in_A, in_B]
        in_A, in_B = inputs
        outA = self.dense_ragged(in_A)
        outB = self.dense_ragged(in_B)
        pooledA = self.pooling_ragged(outA)
        pooledB = self.pooling_ragged(outB)
        # Compute the sum of the pooled outputs, matching model2 behavior
        return pooledA + pooledB


def my_model_function():
    # Return an instance of MyModel
    return MyModel()


def GetInput():
    # Batch size 3 (taken from example data), variable length ragged sequences, 1 feature per element
    # Use tf.ragged.constant to create inputs similar to data_A and data_B

    # Define ragged input A similar to original example in issue
    data_A = tf.ragged.constant([[[2.0], [2.0]],
                                [[3.0]],
                                [[4.0], [5.0], [6.0]]], ragged_rank=1, dtype=tf.float32)
    # Define ragged input B similar to original example in issue
    data_B = tf.ragged.constant([[[4.0], [4.0]],
                                [[6.0]],
                                [[8.0], [10.0], [12.0]]], ragged_rank=1, dtype=tf.float32)

    return [data_A, data_B]

