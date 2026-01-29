# tf.random.uniform((1, 30, 30, 5), dtype=tf.float32)
import tensorflow as tf

class DynamicWeightTest(tf.keras.layers.Layer):
    def __init__(self, channels, select_size, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.select_size = select_size

    def call(self, inputs):
        x = inputs[0]  # input tensor of shape (B, H, W, nb_channels)
        wt = inputs[1] # weights tensor of shape (B, 1, nb_channels, wt_size)

        # Note: tf.nn.conv2d expects wt with shape [filter_height, filter_width, in_channels, out_channels].
        # The weight input has shape (B, 1, nb_channels, wt_size) per batch, but tf.nn.conv2d expects a single weight tensor.
        # We'll assume batch size 1 for simplicity here and slice the 0th element, since original example uses batch=1.
        # Also use strides=(1,1,1,1)
        # Padding 'VALID'
        # This layer is dynamic conv with dynamic weights computed per input.

        conv_out = tf.nn.conv2d(
            x,
            wt[0],           # slice batch dim 0 (shape: (1, nb_channels, wt_size))
            strides=[1, 1, 1, 1],
            padding='VALID'
        )
        return conv_out

class MyModel(tf.keras.Model):
    def __init__(self, nb_channels=5, wt_size=100, in_size=30):
        super().__init__()
        self.nb_channels = nb_channels
        self.wt_size = wt_size
        self.in_size = in_size

        # Instantiate the dynamic conv layer with parameters as per example
        self.dynamic_conv = DynamicWeightTest(channels=wt_size, select_size=nb_channels)

    def call(self, inputs):
        # inputs is a tuple/list: (data_tensor, weight_tensor)
        # data_tensor shape: (1, in_size, in_size, nb_channels)
        # weight_tensor shape: (1, 1, nb_channels, wt_size)

        data, weight = inputs

        # Simply apply the dynamic convolution with dynamic weights for each input
        out = self.dynamic_conv([data, weight])
        return out

def my_model_function():
    # Return an instance of MyModel initialized with standard parameters used in example
    return MyModel(nb_channels=5, wt_size=100, in_size=30)

def GetInput():
    # Return a tuple of inputs matching MyModel expected input:
    # data tensor shape: (1, 30, 30, 5)
    # weight tensor shape: (1, 1, 5, 100)

    input_data = tf.random.uniform((1, 30, 30, 5), dtype=tf.float32)
    input_wt = tf.random.uniform((1, 1, 5, 100), dtype=tf.float32)
    return (input_data, input_wt)

