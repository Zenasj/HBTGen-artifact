# tf.random.uniform((B, 1, 784), dtype=tf.float32) ← inferred input shape from the reshape in call()

import tensorflow as tf

class LaplacianLayer(tf.keras.layers.Layer):
    """
    A custom layer inspired by the described LaplacianLayer.
    This is a guess implementation since the original code is missing.
    It applies a simple learned 1D convolution-like operation along the last dimension.

    Assumptions:
    - n_filters is the output channel dimension.
    - n_input_channels is the input channel dimension.
    - n_params_to_fit is kernel size (1D convolution kernel size).
    - n_dimensions unused in this guess (likely 1 for 1D conv).
    """

    def __init__(self, n_filters, n_input_channels, n_params_to_fit, n_dimensions=1):
        super(LaplacianLayer, self).__init__()
        self.n_filters = n_filters
        self.n_input_channels = n_input_channels
        self.kernel_size = n_params_to_fit
        self.n_dimensions = n_dimensions
        # Initialize a kernel weight matrix, shape: (kernel_size, in_channels, out_channels)
        # This mimics a 1D convolution kernel
        self.kernel = self.add_weight(
            shape=(self.kernel_size, self.n_input_channels, self.n_filters),
            initializer="glorot_uniform",
            trainable=True,
            name="kernel"
        )
        # Optional bias
        self.bias = self.add_weight(
            shape=(self.n_filters,),
            initializer="zeros",
            trainable=True,
            name="bias"
        )

    def call(self, inputs):
        """
        inputs expected shape: (batch, channels_in, length)
        We'll perform a 1D convolution along the last dimension.

        Since TF conv1d expects (batch, length, channels), 
        we need to transpose and then transpose back.

        Output shape: (batch, n_filters, length_out)
        Here, we keep padding='same' to keep length consistent.
        """
        # Inputs shape: (batch, channels_in, length)
        inputs_transposed = tf.transpose(inputs, perm=[0, 2, 1])  # (B, length, channels_in)
        # Perform 1D convolution
        outputs = tf.nn.conv1d(inputs_transposed,
                              filters=self.kernel,
                              stride=1,
                              padding='SAME',
                              data_format='NWC')  # NWC = batch, width, channels
        outputs = outputs + self.bias  # add bias
        outputs = tf.transpose(outputs, perm=[0, 2, 1])  # back to (B, channels_out, length)
        return outputs


class MyModel(tf.keras.Model):
    """
    This model fuses the provided TwoLayerLinalgConv model structure and its
    underlying LaplacianLayer usage inferred from the description.

    The model:
    - Reshapes input to (batch, channels_in=1, length=784)
    - Applies conv1 with learned filters (n_filters)
    - Applies ReLU
    - Applies conv2 with 1 filter
    - Applies ReLU
    - Reshapes and applies dense layer to output logits

    Notes:
    - Output dimension is assumed to be 10 (like MNIST classes).
    - Internal parameters such as kernel size and n_dimensions are inferred/fixed for example.
    """

    def __init__(self):
        super(MyModel, self).__init__()

        # Hard-coded assumptions from original code snippet & MNIST context
        self._n_filters = 4  # example: 4 filters for conv1
        self._n_input_channels = 1
        self._kernel_size = 3  # kernel size 3 for convolutions
        self._n_dimensions = 1  # 1D convolution along length dimension
        self._data_size = 784  # MNIST image flattened size
        self._output_dim = 10  # MNIST has 10 classes

        # Define layers based on inferred LaplacianLayer api
        self.conv1 = LaplacianLayer(
            n_filters=self._n_filters,
            n_input_channels=self._n_input_channels,
            n_params_to_fit=self._kernel_size,
            n_dimensions=self._n_dimensions
        )

        self.conv2 = LaplacianLayer(
            n_filters=1,  # must have 1 filter here for the dense layer in the end
            n_input_channels=self._n_filters,
            n_params_to_fit=self._kernel_size,
            n_dimensions=self._n_dimensions
        )

        # Dense layer for final classification logits
        self.dense = tf.keras.layers.Dense(self._output_dim, activation=None)

    def call(self, inputs):
        """
        inputs: (batch, 784) tensor of floats, flattened MNIST images.

        Workflow:
        - reshape to (batch, channels=1, length=784)
        - conv1 + relu
        - conv2 + relu
        - reshape to (batch, feature_dim) for dense
        - dense (output logits)
        """
        x = tf.reshape(inputs, [-1, self._n_input_channels, self._data_size])  # (B,1,784)
        x = self.conv1(x)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = tf.nn.relu(x)
        # flatten for dense layer
        # conv2 output shape: (B, 1, 784)
        x = tf.reshape(x, [-1, self._data_size])  # (B, 784)
        logits = self.dense(x)  # (B, 10)
        return logits


def my_model_function():
    # Return an instance of MyModel with initialized weights
    return MyModel()


def GetInput():
    # Return a random tensor input for the model: shape (batch, 784)
    # Using batch size 10 as example (typical minibatch size)
    return tf.random.uniform((10, 784), dtype=tf.float32)

