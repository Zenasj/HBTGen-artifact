# tf.random.uniform((64, None), dtype=tf.int32) <- Input shape: batch=64, variable sequence length, single int32 feature per timestep

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.embedding = tf.keras.layers.Embedding(64, 2048)
        # Bidirectional GRU with 2048 units each direction, merge_mode='sum'
        self.bi_gru = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(2048, return_sequences=True), merge_mode="sum"
        )
        # Bidirectional LSTM with 2048 units each direction, merge_mode='sum'
        self.bi_lstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(2048, return_sequences=True), merge_mode="sum"
        )
        self.dense = tf.keras.layers.Dense(10)

    def call(self, inputs, training=None):
        """
        Args:
            inputs: int32 tensor with shape (batch_size, seq_len)
                    - Values >=0 are valid tokens
                    - Values <0 indicate masked positions (padding)
        Returns:
            Tensor of shape (batch_size, seq_len, 10)
        """
        # Create mask based on non-negative input tokens
        mask = tf.math.greater_equal(inputs, 0)

        # Clamp inputs to zero for embedding indexing (to avoid negative indices)
        clamped_inputs = tf.math.maximum(inputs, 0)

        # Embedding lookup
        x = self.embedding(clamped_inputs)  # shape (batch, seq_len, 2048)

        # Run bidirectional GRU with mask
        gru_outputs = self.bi_gru(x, mask=mask)

        # Run bidirectional LSTM with mask
        lstm_outputs = self.bi_lstm(x, mask=mask)

        # Sum outputs and pass through Dense
        combined = gru_outputs + lstm_outputs

        output = self.dense(combined)
        return output

def my_model_function():
    """
    Returns:
        An instance of MyModel initialized.
    """
    # Set global seed for reproducibility to mimic issue scenario
    tf.keras.utils.set_random_seed(42)
    return MyModel()

def GetInput():
    """
    Generates a random input tensor suitable for MyModel.
    
    Based on the issue description:
    - Batch size: 64
    - Sequence lengths variable, masked by input < 0
    - Inputs are integers in range [0, 63], padding represented by -1
    - Each batch element is a sequence padded with -1 (masking)
    
    Returns:
        A (64, 64) int32 tf.Tensor with masked padding (-1).
    """
    batch_size = 64
    max_seq_len = 64

    # Generate a numpy array with shape (64, 64):
    # For row i, values are from 0 to i inclusive, rest are -1 as padding.
    import numpy as np
    data = np.array([[j if j <= i else -1 for j in range(max_seq_len)] for i in range(batch_size)], dtype=np.int32)

    # Convert to tensor
    return tf.convert_to_tensor(data, dtype=tf.int32)

