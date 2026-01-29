# tf.random.uniform((1, 1, 9, 64), dtype=tf.float32)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # DepthwiseConv2D layer setup matching original model from issue
        # kernel_size=(1,9), strides=(1,1), padding='same', channels_last
        self.depthwise_conv = tf.keras.layers.DepthwiseConv2D(
            kernel_size=(1, 9),
            strides=(1, 1),
            padding='same',
            data_format='channels_last',
            depth_multiplier=1,
            activation=None,
            use_bias=True
        )

    def call(self, inputs, training=False):
        # Forward pass applying the single depthwise convolution
        x = self.depthwise_conv(inputs)
        return x

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    # The original input shape is (1, 1, 9, 64) with dtype float32 in training,
    # but for TFLite uint8 quantized inference inputs are often uint8.
    # We use float32 here assuming model expects that (from tf.random.uniform).
    # If quantized inputs are needed, casting and scaling would be required.
    return tf.random.uniform((1, 1, 9, 64), dtype=tf.float32)

