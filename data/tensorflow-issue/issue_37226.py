# tf.random.uniform((None, None, 11), dtype=tf.float32) ← inferred from RaggedTensorSpec(TensorShape([None, None, 11]), tf.float32)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Example LSTM layer and Dense output to match example usage in issue
        # Assuming a sequence model handling ragged inputs with feature size 11
        self.lstm = tf.keras.layers.LSTM(32, return_sequences=False)
        self.dense = tf.keras.layers.Dense(3)  # output size from example

    def call(self, inputs):
        # inputs is expected as a RaggedTensor with shape [batch, None, 11]
        # We convert the ragged input to a dense tensor with padding for LSTM
        # Alternatively, pass the ragged values + row_splits to handle ragged input properly

        if isinstance(inputs, tf.RaggedTensor):
            # Using ragged.to_tensor pads to max length in batch dimension 
            x = inputs.to_tensor(default_value=0.0)
            # Masking could be applied if needed, but keeping simple here
        else:
            # If input is not ragged, assume dense tensor
            x = inputs

        # Now shape should be [batch, seq_len, 11]
        x = self.lstm(x)
        out = self.dense(x)
        return out


def my_model_function():
    # Return an instance of MyModel
    return MyModel()


def GetInput():
    # Returns a RaggedTensor shaped [batch=2, variable length sequences, 11 features]
    # Matching the example in the issue context
    batch_data = [
        [  # first batch sequence with 2 steps (variable length)
            [-0.5]*11,
            [0.2]*11,
        ],
        [  # second batch sequence with 3 steps
            [0.1]*11,
            [0.3]*11,
            [-0.1]*11,
        ],
    ]
    ragged_input = tf.ragged.constant(batch_data, dtype=tf.float32)

    return ragged_input

