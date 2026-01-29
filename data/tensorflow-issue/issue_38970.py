# tf.RaggedTensor with shape (batch_size=3, variable_time_1=None, time_2=20, channels=1)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Input layer with ragged=True to accept RaggedTensor input of shape (None, None, 20, 1)
        self.input_layer = tf.keras.layers.InputLayer(input_shape=(None, 20, 1), ragged=True)

        # TimeDistributed LSTM to process variable-length sequences inside ragged tensor
        self.time_dist_lstm = tf.keras.layers.TimeDistributed(
            tf.keras.layers.LSTM(32, dropout=0.4)
        )
        # Second LSTM layer operates on the output of time distributed LSTM
        self.lstm = tf.keras.layers.LSTM(24)
        # Final dense layer with sigmoid activation for binary classification
        self.dense = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs, training=False):
        # inputs expected: RaggedTensor of shape (batch_size, None, 20, 1)
        x = self.input_layer(inputs)
        # After InputLayer with ragged=True, shape is maintained as RaggedTensor if input was ragged
        # TimeDistributed LSTM can handle ragged inputs on the leading time dimension
        x = self.time_dist_lstm(x, training=training)
        # The output of time_distributed LSTM is a Tensor (batch_size, None, 32)
        # Second LSTM consumes this tensor (with mask automatically handled by ragged dims)
        x = self.lstm(x, training=training)
        # Final output scalar per batch element, shape (batch_size, 1)
        out = self.dense(x)
        return out

def my_model_function():
    # Return an instance of MyModel
    model = MyModel()
    # Compile the model with binary classification loss and optimizer
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model

def GetInput():
    # Return a RaggedTensor input compatible with the model expected input shape:
    # (batch_size=3, variable length ndim1, 20 timesteps, 1 channel)
    # We'll simulate variable length sequences on the second dimension (ragged dimension)
    import numpy as np

    # Here 3 sequences with different lengths along the 2nd dim (ragged dimension)
    # For example: sequence lengths = [4, 12, 84] total=100 as per the example
    # Each internal item is (20, 1)
    values = np.ones((100, 20, 1), dtype=np.float32)
    # Define row splits for ragged dimension 1
    row_splits = [0, 4, 20, 100]
    rt = tf.RaggedTensor.from_row_splits(values, row_splits)

    return rt

