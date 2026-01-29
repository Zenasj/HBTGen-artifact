# tf.random.normal((1, ), dtype=tf.float32) ← The model expects input shape = (1, input_channel) = (1, 32) as single-step input features.

import tensorflow as tf


class CircularBufferLayer(tf.keras.layers.Layer):
    def __init__(self, num_features, buffer_size, stride, **kwargs):
        super().__init__(**kwargs)
        self.num_features = num_features
        self.buffer_size = buffer_size
        self.stride = stride
        # Buffer holds a sliding window of input features over time
        self.buffer = self.add_weight(
            name='buffer',
            shape=(1, buffer_size, self.num_features),
            initializer='zeros',
            trainable=False,
            dtype=tf.float32)
        # Counts how many calls have occurred towards current stride
        self.call_count = self.add_weight(
            name='call_count', shape=(), initializer='zeros', dtype=tf.int32, trainable=False)
        # Counts total calls capped at buffer size
        self.total_call_count = self.add_weight(
            name='total_call_count', shape=(), initializer='zeros', dtype=tf.int32, trainable=False)

    def call(self, inputs, **kwargs):
        # inputs shape assumed (1, num_features) i.e. [batch=1, features]
        inputs = tf.reshape(inputs, [1, 1, self.num_features])  # shape (1, 1, num_features)

        # Update buffer by removing oldest data and adding new inputs (sliding window)
        self.buffer.assign(tf.concat([self.buffer[:, 1:], inputs], axis=1))

        # Update call_count with saturation at stride
        self.call_count.assign(
            tf.cond(
                tf.greater_equal(self.call_count + 1, self.stride),
                true_fn=lambda: self.stride,
                false_fn=lambda: self.call_count + 1
            )
        )

        # Update total_call_count with saturation at buffer_size
        self.total_call_count.assign(
            tf.cond(
                tf.greater_equal(self.total_call_count + 1, self.buffer_size),
                true_fn=lambda: self.buffer_size,
                false_fn=lambda: self.total_call_count + 1
            )
        )

        # If call_count == stride and total_call_count >= buffer_size then reset call_count to 0
        self.call_count.assign(
            tf.cond(
                tf.logical_and(tf.equal(self.call_count, self.stride),
                               tf.greater_equal(self.total_call_count, self.buffer_size)
                              ),
                true_fn=lambda: 0,
                false_fn=lambda: self.call_count
            )
        )

        # Flag indicates if we just reset call_count (window ready)
        flag = tf.equal(self.call_count, 0)

        # Return current buffer and flag
        return [self.buffer, flag]

    def reset(self):
        self.buffer.assign(tf.zeros_like(self.buffer))
        self.call_count.assign(tf.zeros_like(self.call_count))
        self.total_call_count.assign(tf.zeros_like(self.total_call_count))

    def get_config(self):
        config = {
            'buffer': self.buffer,
            'total_call_count': self.total_call_count,
            'call_count': self.call_count
        }
        base_config = super(CircularBufferLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.input_channel = 32
        self.output_channel = 64
        self.kernel_size = 5
        self.stride = 2

        # Circular buffer layer holding a sliding window over inputs
        self.buffer = CircularBufferLayer(
            num_features=self.input_channel,
            buffer_size=self.kernel_size,
            stride=self.stride)

        # Conv1D to process buffered window
        self.conv1d = tf.keras.layers.Conv1D(
            filters=self.output_channel,
            kernel_size=self.kernel_size,
            strides=1,
            use_bias=False,
            padding='valid',
            data_format='channels_last'
        )

    def call(self, inputs, **kwargs):
        # inputs shape: (1, input_channel) i.e. batch=1, features=32
        buffer_out, flag = self.buffer(inputs)  # shape buffer_out (1, kernel_size, input_channel)

        # Conditionally run conv1d when buffer window is ready (flag=True)
        x = tf.cond(
            pred=flag,
            true_fn=lambda: self.conv1d(buffer_out),   # output shape: (1, 1, output_channel)
            false_fn=lambda: tf.zeros([1, 1, self.output_channel], dtype=tf.float32)
        )

        x = tf.reshape(x, [1, self.output_channel])  # shape: (1, output_channel)

        # Return output tensor and the readiness flag
        return [x, flag]

    def reset(self):
        self.buffer.reset()


def my_model_function():
    # Return an instance of MyModel with default initialization
    return MyModel()


def GetInput():
    # Generate input tensor corresponding to one-step input features expected by MyModel
    # Input shape: (1, 32) as a single step of multichannel features
    return tf.random.normal([1, 32], dtype=tf.float32)

