# tf.random.uniform((B, maxlen), dtype=tf.int32) ← input shape is a batch of sequences with length maxlen, integer token ids

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, max_features=20000, maxlen=100, embedding_dim=100, gru_units=64, num_classes=3):
        super().__init__()
        # Embedding layer for integer token inputs
        self.embedding = tf.keras.layers.Embedding(input_dim=max_features, output_dim=embedding_dim, input_length=maxlen)
        # Two Bidirectional GRU layers - first returns sequences, second returns final state
        self.bi_gru_1 = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(units=gru_units, return_sequences=True))
        self.bi_gru_2 = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(units=gru_units, return_sequences=False))
        # Output Dense layer with softmax for multi-class classification (3 classes)
        self.classifier = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, inputs, training=False):
        """
        Forward pass through the model.
        Args:
            inputs: Tensor of shape (batch_size, maxlen), dtype int32 representing tokenized sequences
            training: Bool, if True applies dropout/training specific layers (not used here explicitly)
        Returns:
            Output probabilities with shape (batch_size, num_classes)
        """
        x = self.embedding(inputs)            # (B, maxlen, embedding_dim)
        x = self.bi_gru_1(x)                  # (B, maxlen, gru_units*2)
        x = self.bi_gru_2(x)                  # (B, gru_units*2)
        output = self.classifier(x)           # (B, num_classes)
        return output

def my_model_function():
    """
    Factory function to create an instance of MyModel with typical default parameters.
    """
    # Default max_features and maxlen are typical values inferred from the original code snippet
    max_features = 20000  # Typical vocabulary size for text models
    maxlen = 100          # Sequence length to match Input(shape=(maxlen,))
    embedding_dim = 100
    gru_units = 64
    num_classes = 3
    return MyModel(max_features=max_features, maxlen=maxlen, embedding_dim=embedding_dim,
                   gru_units=gru_units, num_classes=num_classes)

def GetInput():
    """
    Returns a random integer tensor simulating tokenized text input for MyModel.
    Shape: (batch_size, maxlen)
    dtype: tf.int32
    Values between 0 and max_features-1 (vocab indices).
    """
    batch_size = 32       # Typical batch size for testing
    maxlen = 100          # Must match model input sequence length
    max_features = 20000  # Must match vocabulary size of model embedding layer
    # Generate random integers that represent token ids
    return tf.random.uniform(shape=(batch_size, maxlen), minval=0, maxval=max_features, dtype=tf.int32)

