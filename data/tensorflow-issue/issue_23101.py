import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import functools
import tensorflow._api.v1.compat.v2 as tf


class Conv1D(tf.keras.layers.Conv1D):
    """Prevent keras from caching the input shape when building.
    """
    # pylint: disable=no-name-in-module
    from tensorflow.python.keras.utils import conv_utils
    # pylint: enable=no-name-in-module

    def build(self, input_shape):
        """Use the functional convolution operation instead of an instance 
        of the internal TF Convolution class, which stores and reuses the input shape.
        """
        super().build(input_shape)

        if self.padding == 'causal':
            op_padding = 'valid'
        else:
            op_padding = self.padding
        if not isinstance(op_padding, (list, tuple)):
            op_padding = op_padding.upper()

        # pylint: disable=attribute-defined-outside-init
        self._convolution_op = functools.partial(
            tf.nn.convolution,
            strides=self.strides,
            padding=op_padding,
            data_format=self.conv_utils.convert_data_format(
                self.data_format, self.rank + 2
            ),
            dilations=self.dilation_rate
        )
        # pylint: enable=attribute-defined-outside-init