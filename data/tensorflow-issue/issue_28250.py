# tf.random.uniform((B, None), dtype=tf.int64) ← Input shape is (batch_size, variable_sequence_length), dtype int64 for token ids

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        vocab_size = 2
        label_size = 2
        embedding_dim = 100
        lstm_units = 100

        # Embedding layer with trainable=True (simulating the trainable embedding usage)
        self.word_embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, trainable=True)
        # Bidirectional LSTM Encoder
        self.encoder = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_units, return_sequences=True))
        # Classifier layer producing logits for each token position
        self.classifier = tf.keras.layers.Dense(label_size)

    def call(self, word_ids):
        """
        word_ids: tf.Tensor of shape [batch_size, variable_seq_len], dtype int64
        """
        x = self.word_embedding(word_ids)  # shape: [B, seq_len, embedding_dim]
        x = self.encoder(x)                 # shape: [B, seq_len, lstm_units * 2]
        x = self.classifier(x)              # shape: [B, seq_len, label_size]
        return x

def my_model_function():
    # Return an instance of MyModel with trainable embedding layer
    return MyModel()

def GetInput():
    """
    Produce a batch input tensor of shape (batch_size, variable_seq_len), dtype int64, matching vocab_size=2.

    Uses padded batch with variable length sequences to simulate the original issue.
    Here we create batch_size=1 batches with sequences of different lengths.
    We'll just produce one tensor compatible as input to MyModel.
    """
    # We create a batch with batch size 1 and variable sequence length with padding.

    # For demonstration, produce a batch with seq_len=4 (max length),
    # sequence contains token ids in range [0, 1]
    batch_size = 1
    max_seq_len = 4
    
    # Example sequence: [1,1,1,1], dtype int64
    input_tensor = tf.constant([[1, 1, 1, 1]], dtype=tf.int64)

    return input_tensor

