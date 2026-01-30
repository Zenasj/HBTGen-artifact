class MapFlat(layers.Wrapper):
    def __init__(self, layer, **kwargs):
        super().__init__(layer, **kwargs)
        self._supports_ragged_inputs = True

    def build(self, input_shape=None):
        super(MapFlat, self).build([None])

    def call(self, inputs, **kwargs):
        return tf.ragged.map_flat_values(self.layer, inputs)

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape + self.layer.compute_output_shape([None])[1:]

model.add(WithRagged(Dense(2, activation='sigmoid')))

model.add(MapFlat(Dense(2, activation='sigmoid')))

def action_model():

    model = keras.Sequential()

    model.add(Bidirectional(LSTM(101,return_sequences=True),input_shape=(None,101)))
    model.add(Bidirectional(LSTM(25,return_sequences=True)))
    model.add(TimeDistributed(Dense(20,activation='relu')))
    model.add(Dropout(0.4))
    model.add(TimeDistributed(Dense(16,activation='relu')))
    model.add(Dropout(0.4))
    model.add(TimeDistributed((MapFlat(Dense(2, activation='sigmoid')))))

import tensorflow as tf
from keras import backend, layers
from keras.utils.generic_utils import has_arg, register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion


class WithRagged(layers.Wrapper):
    """ Passes ragged tensor to layer that accepts only dense one.

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