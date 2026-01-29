# tf.random.uniform((B, 16), dtype=tf.float32) ← Batch size B inferred as flexible; input feature size 16

import tensorflow as tf
import numpy as np

def complex_uniform_initializer(scale=0.05):
    real_initializer = tf.keras.initializers.RandomUniform(-scale, scale)
    def initializer(shape, dtype=None):
        # If dtype is complex64 or complex128, use float initializer for real and imaginary parts
        # and then combine as a complex tensor.
        if dtype == tf.complex64:
            re_dtype = tf.float32
        elif dtype == tf.complex128:
            re_dtype = tf.float64
        else:
            re_dtype = dtype
        real = real_initializer(shape, re_dtype)
        imag = real_initializer(shape, re_dtype)
        return tf.complex(real, imag)
    return initializer

class ComplexDenseLayer(tf.keras.layers.Layer):
    def __init__(self, out_units, activation=None):
        super().__init__()
        self.out_units = out_units
        self.activation = activation

    def build(self, input_shape):
        inp_units = input_shape[-1]
        initializer = complex_uniform_initializer()
        self.w = self.add_weight(
            shape=[inp_units, self.out_units],
            initializer=initializer,
            dtype=tf.complex64,
            trainable=True)
        self.b = self.add_weight(
            shape=[self.out_units],
            initializer=initializer,
            dtype=tf.complex64,
            trainable=True)

    def call(self, inp):
        # Perform complex matrix multiplication and add bias
        x = tf.einsum('bi,ij->bj', inp, self.w)  # b = batch dim, i and j input/output units
        x = tf.nn.bias_add(x, self.b)
        if self.activation is not None:
            return self.activation(x)
        return x

class MyModel(tf.keras.Model):
    def __init__(self, input_units=16, intermediate_units=128, output_units=16):
        # Fuse the real and complex parts model into one keras.Model subclass compatible with tf2.20 and XLA
        super().__init__()
        self.input_units = input_units
        self.intermediate_units = intermediate_units
        self.output_units = output_units

        # Real dense layers to produce real and imaginary parts
        self.dense_real = tf.keras.layers.Dense(intermediate_units)
        self.dense_imag = tf.keras.layers.Dense(intermediate_units)

        # Complex dense layer with custom complex weights and optional activation
        # Activation used: w * conj(w), representing magnitude squared
        self.complex_layer = ComplexDenseLayer(
            intermediate_units,
            activation=lambda w: w * tf.math.conj(w))

        # Output dense layer on real result
        self.output_dense = tf.keras.layers.Dense(output_units)

    def call(self, inputs):
        """
        inputs: Tensor of shape (batch_size, input_units), dtype float32
        Process:
        - Feed inputs through two parallel real dense layers to get real and imag parts
        - Compose complex tensor from these
        - Pass through complex dense layer with magnitude squared activation
        - Reduce to real by taking real part
        - Final dense layer for output
        """
        xreal = self.dense_real(inputs)
        ximag = self.dense_imag(inputs)
        xcomplex = tf.cast(xreal, tf.complex64) + 1j * tf.cast(ximag, tf.complex64)
        x = self.complex_layer(xcomplex)
        x = tf.math.real(x)
        out = self.output_dense(x)
        return out

def my_model_function():
    # Instantiate and return the model with default sizes from the issue
    return MyModel(input_units=16, intermediate_units=128, output_units=16)

def GetInput():
    # Return a batch of random float32 input tensors with shape (batch_size, input_units)
    # Use batch size 10 as in the example
    batch_size = 10
    input_units = 16
    # Uniform random inputs between 0 and 1 as per original example
    return tf.random.uniform((batch_size, input_units), dtype=tf.float32)

