import tensorflow as tf
from tensorflow import keras

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
    
    def call(self, input):
        if self.pad_type in ['symmetric', 'reflect']:
            input_rows = tf.shape(input)[2]
            filter_rows = self.kernel_size[0]
            out_rows = (input_rows + self.strides[0] - 1) // self.strides[0]
            padding_rows = tf.maximum(0, (out_rows - 1) * self.strides[0] +
                                      (filter_rows - 1) * self.dilation_rate[0] + 1 - input_rows)
            rows_odd = tf.mod(padding_rows, 2)

            input_cols = tf.shape(input)[3]
            filter_cols = self.kernel_size[1]
            out_cols = (input_cols + self.strides[1] - 1) // self.strides[1]
            padding_cols = tf.maximum(0, (out_cols - 1) * self.strides[1] +
                                      (filter_cols - 1) * self.dilation_rate[1] + 1 - input_cols)
            cols_odd = tf.mod(padding_cols, 2)

            output = tf.pad(input, [[0, 0], [padding_rows // 2, padding_rows // 2 + rows_odd],
                            [padding_cols // 2, padding_cols // 2 + cols_odd], [0, 0]], mode=self.pad_type)

            return K.conv2d(output,
                            self.kernel,
                            strides=self.strides,
                            padding='valid',
                            data_format=self.data_format,
                            dilation_rate=self.dilation_rate)

        elif self.pad_type in ['same', 'valid']:
            return K.conv2d(input,
                            self.kernel,
                            strides=self.strides,
                            padding=self.padding,
                            data_format=self.data_format,
                            dilation_rate=self.dilation_rate)

inputs = Input(shape=(None, None, 1))
x = inputs
x = CustomConv2D(filters=1, kernel_size=(k, k), padding='same')(x)
model = Model(inputs=inputs, outputs=x)
optimizer = Adam(config.learning_rate, config.momentum)
model.compile(loss='mean_squared_error',
                         optimizer=optimizer)
model.fit_generator(batch_gen, epochs=100, steps_per_epoch=10)