# tf.zeros([10, 5], dtype=tf.complex64) ← Input is a batch of 10 vectors with 5 complex features

import tensorflow as tf

class MyModel(tf.keras.Model):
    """
    This model replicates the described custom dense layer that supports complex64 inputs.
    It holds a weight matrix (kernel) as a float variable but casts it to complex64 during multiplication.
    The model supports input tensors of shape (batch_size, 5) with dtype complex64,
    and outputs a tensor of shape (batch_size, 10) with dtype complex64.
    """
    def __init__(self, num_outputs=10):
        super(MyModel, self).__init__()
        self.num_outputs = num_outputs

    def build(self, input_shape):
        # input_shape should be (batch_size, 5)
        # We create kernel weights as float32 (real) since Keras 3 currently does not support complex dtypes
        input_dim = int(input_shape[-1])
        self.kernel = self.add_weight(
            name="kernel",
            shape=(input_dim, self.num_outputs),
            dtype=tf.float32,
            initializer="glorot_uniform",
            trainable=True,
        )

    def call(self, inputs):
        # Cast kernel to complex64 on the fly
        kernel_complex = tf.cast(self.kernel, tf.complex64)
        # Perform matrix multiplication with complex inputs
        return tf.matmul(inputs, kernel_complex)


def my_model_function():
    """
    Returns an instance of MyModel initialized with 10 outputs.
    """
    return MyModel(num_outputs=10)


def GetInput():
    """
    Generates a random complex64 tensor input compatible with MyModel.
    Input shape: (10, 5)
    Real and imaginary parts are sampled uniformly between 0 and 1.
    """
    real_part = tf.random.uniform((10, 5), dtype=tf.float32)
    imag_part = tf.random.uniform((10, 5), dtype=tf.float32)
    complex_input = tf.complex(real_part, imag_part)
    return complex_input

