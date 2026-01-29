# tf.random.normal((B, 32, 32, 3), dtype=tf.float32) ← inferred input shape/tensor for MyModel's call function

import tensorflow as tf
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework.func_graph import def_function

class MyModel(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Conv2D layer: kernel 7x7, 64 filters, stride 2, valid padding, channels_last
        self.conv2 = tf.keras.layers.Conv2D(
            64, (7, 7), strides=(2, 2), padding='valid', data_format='channels_last')
        # Dense layer with 10 outputs + softmax activation
        self.dense = tf.keras.layers.Dense(10, activation='softmax')

    @def_function.function(input_signature=[
        tensor_spec.TensorSpec(shape=[None, 32, 32, 3], dtype=tf.float32)
    ])
    def call(self, inputs, training=True, **kwargs):
        # Apply conv2 layer - this caused calibration errors in INT8 TensorRT conversion per issue
        x = self.conv2(inputs)
        # Reshape output to collapse spatial dims, flatten last channel dimension preserved
        x = tf.reshape(x, [-1, x.shape[-1]])
        # Apply dense layer
        x = self.dense(x)
        return x

def my_model_function():
    """
    Instantiate and return an instance of MyModel.
    """
    return MyModel()

def GetInput():
    """
    Return a random tensor input matching MyModel's input expectation.
    The input shape is (batch_size=1, height=32, width=32, channels=3) and dtype float32.
    Yielded as a tuple to satisfy TensorRT int8 calibration input_fn requirement.
    """
    # According to comments on issue, input_fn used during INT8 calibration must yield tuples.
    # Batch size = 1 (typical for calibration).
    # Use tf.random.normal with correct shape and dtype.
    # Yield as tuple (not bare tensor) to avoid calibration errors reported in the issue.
    return (tf.random.normal((1, 32, 32, 3), dtype=tf.float32),)

