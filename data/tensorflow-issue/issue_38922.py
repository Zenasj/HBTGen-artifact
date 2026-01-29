# tf.random.uniform((B, T), dtype=tf.int32) ← Input shape inference: batch size B, sequence length T (padded sequences)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, max_features=20000, embedding_dim=128, lstm_units=128, dropout_rate=0.2):
        super().__init__()
        self.embedding = tf.keras.layers.Embedding(input_dim=max_features, output_dim=embedding_dim)
        # Create LSTM layer with dropout and recurrent_dropout as per issue code
        self.lstm = tf.keras.layers.LSTM(lstm_units,
                                         dropout=dropout_rate,
                                         recurrent_dropout=dropout_rate,
                                         return_sequences=False)
        self.dense = tf.keras.layers.Dense(1, activation='sigmoid')

    @tf.function(jit_compile=True)
    def call(self, inputs, training=False):
        x = self.embedding(inputs)
        x = self.lstm(x, training=training)
        output = self.dense(x)
        return output

    def get_dropout_masks(self, inputs, training=True):
        """
        Get dropout and recurrent dropout masks from the LSTMCell inside the LSTM layer.
        Note: The masks are regenerated on each call by Keras LSTMCell.
        This function replicates the attempt from the issue to inspect the masks.

        Assumptions:
        - The LSTM layer has 1 cell, accessible via self.lstm.cell
        - The count=4 is from example; it returns 4 mask tensors (inputs + recurrent parts)
        - The masks shape depends on input batch size and layer units.
        """
        # To keep it consistent with the issue sample:
        # Inputs shape: (batch_size, time_steps)
        # We pass inputs and call cell method get_dropout_mask_for_cell
        # This returns a list of masks — one for each dropout location (input, recurrent).
        # Use lstm.cell.get_dropout_mask_for_cell
        return self.lstm.cell.get_dropout_mask_for_cell(inputs, training=training, count=4)

def my_model_function():
    # Return an instance of MyModel with default params matching example
    return MyModel()

def GetInput():
    """
    Generate a valid input tensor of shape (batch_size, time_steps) with integer word indices.
    Assumptions from issue:
    - max_features=20000 (vocab size)
    - maxlen=80 (sequence length)
    - batch size = 1 (like x_sample in example)
    We generate random integers in [1, max_features).
    """
    batch_size = 1
    maxlen = 80
    max_features = 20000
    # Random integer indices shape: (batch_size, maxlen), dtype int32 for embedding layer indexing
    return tf.random.uniform((batch_size, maxlen), minval=1, maxval=max_features, dtype=tf.int32)

