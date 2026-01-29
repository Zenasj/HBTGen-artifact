# tf.random.uniform((B, None, 101), dtype=tf.float32)  # Assumed input shape (batch_size, variable_seq_len, feature_dim=101)

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import backend
from tensorflow.keras.utils import has_arg

class WithRagged(layers.Wrapper):
    """
    Passes ragged tensor to layer that accepts only dense one.

    Arguments:
      layer: The `Layer` instance to be wrapped.
    """

    def __init__(self, layer, **kwargs):
        super(WithRagged, self).__init__(layer, **kwargs)
        self.input_spec = layer.input_spec
        self.supports_masking = layer.supports_masking
        self._supports_ragged_inputs = True

        if not isinstance(layer, layers.Layer):
            raise ValueError(
                'Please initialize `WithRagged` layer with a '
                '`Layer` instance. You passed: {input}'.format(input=layer))

    def build(self, input_shape=None):
        if not self.layer.built:
            self.layer.build(input_shape)
        self.input_spec = self.layer.input_spec

        zero = '' if self.layer.dtype == tf.string else 0
        self.masking_layer = layers.Masking(mask_value=zero)

        super(WithRagged, self).build()

    def call(self, inputs, **kwargs):
        # Pass only supported kwargs to wrapped layer
        layer_kwargs = {}
        for key in kwargs.keys():
            if has_arg(self.layer.call, key):
                layer_kwargs[key] = kwargs[key]

        inputs_dense, row_lengths = backend.convert_inputs_if_ragged(inputs)
        inputs_dense = self.masking_layer(inputs_dense)
        outputs_dense = self.layer.call(inputs_dense, **layer_kwargs)
        outputs = backend.maybe_convert_to_ragged(row_lengths is not None, outputs_dense, row_lengths)

        return outputs

    def compute_output_shape(self, input_shape):
        return self.layer.compute_output_shape(input_shape)

    def compute_mask(self, inputs, mask=None):
        return self.layer.compute_mask(inputs, mask)

    def get_config(self):
        return super(WithRagged, self).get_config()


class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()

        # Bidirectional LSTM layers supporting variable length sequences
        self.bidir_lstm1 = layers.Bidirectional(layers.LSTM(101, return_sequences=True), input_shape=(None, 101))
        self.bidir_lstm2 = layers.Bidirectional(layers.LSTM(25, return_sequences=True))

        # TimeDistributed dense layers wrapped with WithRagged for supporting ragged tensors
        # Activation relu for intermediate layers
        self.td_dense1 = layers.TimeDistributed(layers.Dense(20, activation='relu'))
        self.dropout1 = layers.Dropout(0.4)
        self.td_dense2 = layers.TimeDistributed(layers.Dense(16, activation='relu'))
        self.dropout2 = layers.Dropout(0.4)

        # Final output TimeDistributed Dense layer wrapped with WithRagged to produce ragged output
        self.td_dense3 = layers.TimeDistributed(WithRagged(layers.Dense(2, activation='sigmoid')))

    def call(self, inputs, training=False):
        """
        Forward pass for variable-length sequence input.
        Inputs can be tf.RaggedTensor or dense Tensor with padding.

        Args:
          inputs: tf.Tensor of shape (batch_size, variable_seq_len, 101) or RaggedTensor
          training: Boolean flag for dropout layers

        Returns:
          RaggedTensor output of shape (batch_size, variable_seq_len, 2) with sigmoid activations
        """

        # Process through BiLSTM layers (support ragged input)
        x = self.bidir_lstm1(inputs)
        x = self.bidir_lstm2(x)

        x = self.td_dense1(x)
        x = self.dropout1(x, training=training)
        x = self.td_dense2(x)
        x = self.dropout2(x, training=training)
        # Final layer outputs ragged tensor because of WithRagged wrapper
        x = self.td_dense3(x)
        return x


def my_model_function():
    """
    Construct and return an instance of MyModel.

    Returns:
      MyModel instance ready for training or inference.
    """
    return MyModel()


def GetInput():
    """
    Generate a random ragged tensor input batch compatible with MyModel.

    Assumptions:
    - Feature dimension = 101
    - Batch size = 5 (arbitrary)
    - Sequence lengths variable between 5 to 10 (random)

    Returns:
      tf.RaggedTensor of shape (batch_size, variable_seq_len, 101), dtype=tf.float32
    """
    batch_size = 5
    feature_dim = 101
    import numpy as np

    # Random sequence lengths between 5 and 10 for each example
    seq_lengths = np.random.randint(low=5, high=11, size=(batch_size,))
    # Create RaggedTensor value list
    values = []
    for length in seq_lengths:
        values.append(tf.random.uniform((length, feature_dim), dtype=tf.float32))
    ragged_input = tf.ragged.stack(values)

    return ragged_input

