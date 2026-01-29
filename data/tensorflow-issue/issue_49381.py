# tf.random.normal((1, 1, 227, 227)) with channels_first (NCHW) for Conv2D input example
# and tf.random.uniform((1, 4, 20)) for LSTM input example (batch, steps, features)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Conv2D model - corrected to channels_last (NHWC) input since TFLite does not support NCHW
        # Input shape for this conv model will be (batch, height, width, channels) = (1, 227, 227, 1)
        # We reduce channels to 1 here because the original code tried data_format='channels_first' with 4 dims, but TFLite expects NHWC.
        # To align with the issue discussion: NCHW is not supported by TFLite, only NHWC.
        self.conv_model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(227, 227, 1)),
            tf.keras.layers.Conv2D(96, 11, strides=4, activation='relu', dilation_rate=(1,1), groups=1, data_format='channels_last')
        ])

        # LSTM Model: input shape (batch, steps, features) = (None, None, 20)
        # This matches TFLite supported NSD format for LSTM inputs
        self.lstm_model = tf.keras.Sequential([
            tf.keras.layers.LSTM(128, return_sequences=True, input_shape=(None, 20)),
            tf.keras.layers.Dense(10)
        ])

    def call(self, inputs):
        # inputs: tuple of (conv_input, lstm_input)
        conv_input, lstm_input = inputs
        
        conv_out = self.conv_model(conv_input)  # shape: (batch, h, w, filters), h and w depend on conv params
        lstm_out = self.lstm_model(lstm_input)  # shape: (batch, steps, 10)
        
        # For demonstration, output a dict of both outputs
        return {
            'conv_output': conv_out,
            'lstm_output': lstm_out
        }

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Provide inputs suitable for MyModel.call()
    # Conv input: NHWC format since TFLite only supports NHWC, shape (1, 227, 227, 1)
    conv_input = tf.random.normal((1, 227, 227, 1), dtype=tf.float32)

    # LSTM input: batch=1, steps=4, features=20 as used in the issue example
    lstm_input = tf.random.uniform((1, 4, 20), dtype=tf.float32)

    # Return inputs as a tuple (conv_input, lstm_input) matching the forward signature
    return (conv_input, lstm_input)

