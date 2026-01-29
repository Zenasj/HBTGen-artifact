# tf.random.uniform((1, 11, 1, 1, 1), dtype=tf.float32)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # According to the issue: 
        # 1) strides must be > 0 (stride=0 causes crash).
        # 2) dtype float64 causes device not found error.
        # 3) pool_size and strides must be compatible with input shape for VALID padding.
        #
        # So here we correct the strides from [0,21.0,1] to valid positive integers,
        # and use dtype float32 instead of float64.
        #
        # Input shape: (1,11,1,1,1) with data_format="channels_last"
        # which corresponds to (batch, depth, height, width, channels)
        # The middle spatial dims are small, so pick strides <= input dims.
        #
        # We'll set:
        # pool_size = [3,3,3]
        # strides = [1,1,1]  # all positive valid strides

        pool_size = [3, 3, 3]
        strides = [1, 1, 1]
        padding = "valid"
        data_format = "channels_last"

        self.avgpool3d = tf.keras.layers.AveragePooling3D(
            pool_size=pool_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dtype=tf.float32
        )

    @tf.function(jit_compile=True)
    def call(self, inputs):
        # Directly apply the avgpool3d layer
        return self.avgpool3d(inputs)


def my_model_function():
    # Returns an instance of MyModel with fixed, valid parameters
    return MyModel()


def GetInput():
    # Generate a float32 input that matches the expected shape and dtype
    # Input shape as per the issue: (1, 11, 1, 1, 1)
    # Use tf.random.uniform with range matching the example in the issue
    return tf.random.uniform([1, 11, 1, 1, 1], minval=-1.0, maxval=0.776402213, dtype=tf.float32)

