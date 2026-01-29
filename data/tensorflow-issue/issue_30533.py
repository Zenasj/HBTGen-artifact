# tf.random.uniform((B, T), dtype=tf.int32) ← Input shape inferred roughly as (batch_size, time_steps)
# The input is integer token indices, padded with zeros, with masking handled separately.

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Masking layer to ignore zero-padding tokens
        self.masking = tf.keras.layers.Masking(mask_value=0)
        # Embedding layer with vocabulary size 300 and embedding dimension 2, trainable=True as per original model
        self.embedding = tf.keras.layers.Embedding(300, 2, trainable=True)
        # LSTM layer with 2 units, "tanh" activation to use cuDNN implementation (avoids Unsupported Type: 21 error)
        # Returns sequences and no states (for compatibility with original code)
        self.encoder = tf.keras.layers.LSTM(2,
                                            activation="tanh",
                                            return_sequences=True,
                                            return_state=False)

    @tf.function(jit_compile=True)
    def call(self, inputs):
        # inputs expected as integer token indices padded with zeros
        x = self.masking(inputs)
        x = self.embedding(x)
        x = self.encoder(x)
        # The original code returns output[0] but encoder returns tensor directly since return_state=False
        # So just return x directly
        return x

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Generate a random integer tensor simulating token indices including padding zeros
    # Assumptions:
    # - batch size = 1 (as in original example)
    # - sequence length = 22 (from input_questions shape)
    # - token indices range from 0 (padding) to 299 (vocab size)
    batch_size = 1
    sequence_length = 22
    vocab_size = 300

    # Generate a random integer tensor with zeros included for padding
    # We generate tokens in [1, vocab_size-1] plus some zeros
    import numpy as np
    np.random.seed(0)

    # For simplicity, produce mostly non-zero tokens, with trailing zeros as padding
    tokens = np.random.randint(1, vocab_size, size=(batch_size, sequence_length))
    # Zero-pad last few positions (simulate padding tokens)
    tokens[0, 15:] = 0

    return tf.convert_to_tensor(tokens, dtype=tf.int32)

