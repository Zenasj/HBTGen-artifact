# tf.random.uniform((B, 32, 32, 3), dtype=tf.float32) for TimeDistributed LSTM branch
# tf.random.uniform((B, 32, 32, 3), dtype=tf.float32) for SeparableConv2D branch (using input shape from chunk 2/3)

import tensorflow as tf
from tensorflow.keras import layers

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # TimeDistributed LSTM branch (from chunk 1)
        # Assumptions:
        # - Input shape: [32, 32, 3]
        # - The LSTM layer inside TimeDistributed expects a sequence dimension at axis=1,
        #   so the input is presumably: (batch, timesteps=32, features=32*3) or simply (batch, 32, 32, 3)
        #   The original code wrapped LSTM with TimeDistributed, implying input shape (batch, ?, 32, 3).
        #   But from the Input shape, it does not explicitly mention a sequence dimension. 
        #   To make it consistent, we will treat input shape as (batch, 32, 32, 3),
        #   then apply TimeDistributed over axis 1 or 2. To be faithful, just replicate the original definition.
        #   Original code used TimeDistributed(LSTM(4,...), name='01_time_distributed')(layer_0)
        
        self.input_shape_td = (32, 32, 3)
        self.input_shape_sc = (32, 32, 3)
        
        # Build TimeDistributed LSTM branch components
        lstm_layer = layers.LSTM(
            units=4,
            activation='softmax',
            recurrent_activation='elu',
            use_bias=False,
            kernel_initializer='random_uniform',
            recurrent_initializer='random_uniform',
            bias_initializer='random_uniform',
            unit_forget_bias=False,
            return_sequences=True,
            recurrent_dropout=0.0,
            dropout=0.0,
            implementation=1,
            stateful=False,
            unroll=False,
            name='lstm_layer'
        )
        self.td_lstm = layers.TimeDistributed(lstm_layer, name='01_time_distributed')
        self.td_flatten = layers.Flatten(name='02_flatten')
        self.td_dense = layers.Dense(
            units=10,
            activation='linear',
            use_bias=False,
            kernel_initializer='random_uniform',
            bias_initializer='random_uniform',
            name='03_dense'
        )
        self.td_reshape = layers.Reshape(target_shape=[10], name='04_reshape')
        
        # Build SeparableConv2D branch components (from chunk 2)
        self.sc_input_shape = (32, 32, 3)  # chunk 2 shows input shape (32, 32, 3) for separable conv
        self.sc_sep_conv2d = layers.SeparableConv2D(
            filters=4,
            kernel_size=[27, 27],
            strides=[1, 1],
            padding='same',
            data_format='channels_last',
            dilation_rate=[1, 1],
            depth_multiplier=5,
            activation='sigmoid',
            use_bias=True,
            depthwise_initializer='random_uniform',
            pointwise_initializer='random_uniform',
            bias_initializer='random_uniform',
            name='01_separable_conv2d'
        )
        self.sc_flatten = layers.Flatten(name='02_flatten')
        self.sc_dense = layers.Dense(
            units=10,
            activation='linear',
            use_bias=False,
            kernel_initializer='random_uniform',
            bias_initializer='random_uniform',
            name='03_dense'
        )
        self.sc_reshape = layers.Reshape(target_shape=[10], name='04_reshape')

    def call(self, inputs, training=False):
        # inputs shape assumed to be (batch_size, 32, 32, 3)
        # Apply TimeDistributed LSTM branch
        # TimeDistributed expects a sequence dimension: from original chunk 1 input is (batch_size, 32, 32, 3)
        # TimeDistributed applies layer on a specific axis,
        # here axis=1 (the 2nd dim with size 32) treated as 'time steps'
        # So apply td_lstm to inputs directly
        td_out = self.td_lstm(inputs, training=training)  # shape expected: [batch, time_steps=32, units=4]
        td_out = self.td_flatten(td_out)                 # flatten
        td_out = self.td_dense(td_out)
        td_out = self.td_reshape(td_out)

        # Apply SeparableConv2D branch
        sc_out = self.sc_sep_conv2d(inputs, training=training)
        sc_out = self.sc_flatten(sc_out)
        sc_out = self.sc_dense(sc_out)
        sc_out = self.sc_reshape(sc_out)

        # Comparing the two outputs elementwise, producing a boolean tensor
        # Using relative tolerance = 1e-6 and absolute tolerance = 1e-6 similar to numerical comparisons
        atol = 1e-6
        rtol = 1e-6
        diff = tf.abs(td_out - sc_out)
        tolerance = atol + rtol * tf.abs(sc_out)
        comparison = diff <= tolerance

        # Return the boolean comparison tensor to indicate per-element closeness
        return comparison

def my_model_function():
    return MyModel()

def GetInput():
    # Batch size: 4 arbitrarily chosen
    batch_size = 4
    # Input shape matches the original inputs: (batch, 32, 32, 3)
    # Using uniform random values as typical inputs for reproduction of model behavior
    input_tensor = tf.random.uniform(shape=(batch_size, 32, 32, 3), dtype=tf.float32)
    return input_tensor

