# tf.random.uniform((2, 2, 2, 5, 1), dtype=tf.float32)  # (batch, time_steps, channels, height, width) with channels_first

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Parameters inferred from the issue comments and example
        filters = 2
        kernel_size = [4, 2]
        strides = [1, 1]
        padding = "same"  # changed from "valid" to "same" to avoid zero spatial output shape
        data_format = "channels_first"
        dilation_rate = [1, 1]
        activation = "tanh"
        recurrent_activation = "hard_sigmoid"
        use_bias = False
        # Initializers and constraints are left as default/None because original example did not set them explicitly
        unit_forget_bias = True
        return_sequences = True  # Important for gradient computation and aligned output shape
        return_state = False
        go_backwards = True  # as in the original example to match behavior
        stateful = False
        dropout = 0.0
        recurrent_dropout = 0.0

        self.conv_lstm = tf.keras.layers.ConvLSTM2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            recurrent_activation=recurrent_activation,
            use_bias=use_bias,
            unit_forget_bias=unit_forget_bias,
            return_sequences=return_sequences,
            return_state=return_state,
            go_backwards=go_backwards,
            stateful=stateful,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
        )

    def call(self, inputs):
        # inputs shape: (batch, time, channels, height, width) with channels_first
        # directly forward through ConvLSTM2D layer
        return self.conv_lstm(inputs)

def my_model_function():
    # Return an initialized instance of MyModel
    return MyModel()

def GetInput():
    # Generate a random input tensor matching the input shape used in the examples
    # Shape: [batch_size=2, time_steps=2, channels=2, height=5, width=1]
    # Channels_first means channel dimension is the 3rd dimension
    return tf.random.uniform(
        (2, 2, 2, 5, 1),
        minval=1.75,
        maxval=2.90,
        dtype=tf.float32
    )

