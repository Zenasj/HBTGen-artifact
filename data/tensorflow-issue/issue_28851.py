# tf.random.uniform((B, H, W, C), dtype=...) ← assuming input shape is (batch, height, width, channels) where height and width can be None

import tensorflow as tf
from tensorflow.keras.layers import Conv2D
from tensorflow.keras import backend as K

class MyModel(tf.keras.Model):
    def __init__(self, filters=1, kernel_size=(3,3), padding='same', **kwargs):
        super(MyModel, self).__init__(**kwargs)
        # Using the CustomConv2D logic provided, integrated as a subcomponent
        self.custom_conv = CustomConv2D(filters=filters, kernel_size=kernel_size, padding=padding)

    def call(self, inputs):
        return self.custom_conv(inputs)
        
class CustomConv2D(Conv2D):
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 dilation_rate=(1, 1),
                 activation=None,
                 use_bias=True,
                 kernel_initializer=tf.keras.initializers.TruncatedNormal(0.0, 0.01),
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):

        self.pad_type = padding.lower()
        # Conv2D expects padding in ['valid', 'same'], fallback to 'valid' if pad_type is custom
        super(CustomConv2D, self).__init__(filters,
                                           kernel_size,
                                           strides,
                                           self.pad_type if self.pad_type in ['valid', 'same'] else 'valid',
                                           'channels_last',
                                           dilation_rate,
                                           activation,
                                           use_bias,
                                           kernel_initializer,
                                           bias_initializer,
                                           activity_regularizer,
                                           kernel_constraint,
                                           bias_constraint,
                                           **kwargs)

    def call(self, inputs):
        # Support for 'symmetric' or 'reflect' padding implemented manually
        if self.pad_type in ['symmetric', 'reflect']:
            input_rows = tf.shape(inputs)[1]
            filter_rows = self.kernel_size[0]
            out_rows = (input_rows + self.strides[0] - 1) // self.strides[0]
            padding_rows = tf.maximum(
                0,
                (out_rows - 1) * self.strides[0] +
                (filter_rows - 1) * self.dilation_rate[0] + 1 - input_rows
            )
            rows_odd = tf.math.mod(padding_rows, 2)

            input_cols = tf.shape(inputs)[2]
            filter_cols = self.kernel_size[1]
            out_cols = (input_cols + self.strides[1] - 1) // self.strides[1]
            padding_cols = tf.maximum(
                0,
                (out_cols - 1) * self.strides[1] +
                (filter_cols - 1) * self.dilation_rate[1] + 1 - input_cols
            )
            cols_odd = tf.math.mod(padding_cols, 2)

            padded = tf.pad(inputs,
                            [[0, 0],
                             [padding_rows // 2, padding_rows // 2 + rows_odd],
                             [padding_cols // 2, padding_cols // 2 + cols_odd],
                             [0, 0]],
                            mode=self.pad_type.upper())

            return K.conv2d(padded,
                            self.kernel,
                            strides=self.strides,
                            padding='valid',
                            data_format=self.data_format,
                            dilation_rate=self.dilation_rate)

        elif self.pad_type in ['same', 'valid']:
            # Direct call to backend conv2d which is differentiable
            return K.conv2d(inputs,
                            self.kernel,
                            strides=self.strides,
                            padding=self.padding,
                            data_format=self.data_format,
                            dilation_rate=self.dilation_rate)
        else:
            # Fallback: treat as 'valid' padding but no manual pad
            return K.conv2d(inputs,
                            self.kernel,
                            strides=self.strides,
                            padding='valid',
                            data_format=self.data_format,
                            dilation_rate=self.dilation_rate)

def my_model_function():
    # Return an instance of MyModel with default params compatible with the example usage
    # Here default filters=1, kernel_size=(3,3), padding='same'
    return MyModel()

def GetInput():
    # Return a single input tensor with shape (batch=1, height=32, width=32, channels=1),
    # matching a typical Conv2D input. Height and width can be flexible,
    # but fixed size helps demonstrate functionality.
    return tf.random.uniform(shape=(1, 32, 32, 1), dtype=tf.float32)

