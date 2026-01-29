# tf.random.uniform((B, None), dtype=tf.int32) ← input shape batch size by variable length sequence

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Embedding layer: input_dim=5000 (vocab size), output_dim=64 (embedding dim)
        self.embedding = tf.keras.layers.Embedding(input_dim=5000, output_dim=64)
        # Dense layer projecting to vocab logits (5000)
        self.dense = tf.keras.layers.Dense(5000)
    
    def call(self, inputs, training=False):
        # inputs: shape (batch_size, seq_length), dtype int32 (token indices)
        x = self.embedding(inputs)   # shape (batch_size, seq_length, 64)
        x = self.dense(x)            # shape (batch_size, seq_length, 5000)
        return x

def my_model_function():
    # Return an instance of MyModel, compiled with Adam optimizer and sparse categorical crossentropy loss
    model = MyModel()
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam()
    )
    return model

def GetInput():
    # Return a random batched integer tensor of shape (batch_size, variable_length)
    # Matching the input of MyModel: variable length sequences of int32 tokens from [1, 5000)
    batch_size = 16  # example batch size
    max_length = 64  # max length for sequences
    # Random lengths for sequences in batch (simulate variable length sequences)
    lengths = tf.random.uniform(shape=(batch_size,), minval=1, maxval=max_length+1, dtype=tf.int32)
    # Build ragged batch of sequences with random token ids
    # To create a dense padded tensor, pad with zeros
    sequences = []
    for length in lengths.numpy():
        seq = tf.random.uniform(shape=(length,), minval=1, maxval=5000, dtype=tf.int32)
        sequences.append(seq)
    ragged_input = tf.ragged.constant(sequences, dtype=tf.int32)
    # Pad to max_length (64) for dense tensor input (0 padding)
    dense_input = ragged_input.to_tensor(default_value=0, shape=[batch_size, max_length])
    return dense_input

