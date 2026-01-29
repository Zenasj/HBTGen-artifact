# tf.random.uniform((B=4, H=100, W=100, C=3), dtype=tf.float32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, num_layers=3):
        super(MyModel, self).__init__()
        self.num_layers = num_layers
        # We create a single ConvLayer instance that will internally create weights once.
        # The original issue applied the same layer multiple times on the input, which is uncommon.
        # Here we mimic the same behavior for demonstration.
        self.conv_layer = self.ConvLayer()

    class ConvLayer(tf.keras.layers.Layer):
        def __init__(self):
            super(MyModel.ConvLayer, self).__init__()
            # Define weights in build() as recommended by Keras best practices
            self.weights = None

        def build(self, input_shape):
            # input_shape expected: (batch, height, width, channels)
            kernel_shape = (3, 3, input_shape[-1], 16)  # (filter_height, filter_width, in_channels, out_channels)
            initializer = tf.keras.initializers.TruncatedNormal(stddev=1.0)
            self.weights = self.add_weight(
                name='weights',
                shape=kernel_shape,
                initializer=initializer,
                trainable=True,
            )
            super(MyModel.ConvLayer, self).build(input_shape)

        def call(self, inputs):
            # Apply conv2d with stride 1 and SAME padding
            conv = tf.nn.conv2d(inputs, self.weights, strides=[1,1,1,1], padding='SAME')
            return conv

    def call(self, inputs):
        """
        Mimics the original pattern from the issue:
        The same ConvLayer instance is applied multiple times to the same input tensor,
        collecting outputs in a list, then concatenating them along axis=0 (batch dimension).
        """
        values = []
        for _ in range(self.num_layers):
            values.append(self.conv_layer(inputs))
        # Concatenate outputs along batch dimension (axis=0)
        stacked_values = tf.concat(values, axis=0)
        return stacked_values


def my_model_function():
    # Return instance of MyModel with default 3 layer calls to mimic original behavior
    return MyModel(num_layers=3)


def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    # Input shape is batch of 4, 100x100 RGB images as per original input_fn batching.
    # Batch size 4 to match the batching in input_fn noted in the issue.
    shape = (4, 100, 100, 3)
    return tf.random.uniform(shape, dtype=tf.float32)

