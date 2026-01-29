# tf.random.uniform((B, max_length), dtype=tf.int32) ← Input shape is (batch_size, max_length)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, n_words, n_tags, max_length, embedding_size=80, hidden_state_encoder_size=100, dropout_rate=0.5):
        super().__init__()
        self.embedding = tf.keras.layers.Embedding(
            input_dim=n_words + 1,
            output_dim=embedding_size,
            input_length=max_length,
            name="Embedding"
        )
        self.encoder = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                units=hidden_state_encoder_size,
                return_sequences=True,
                dropout=dropout_rate,
                name="LSTM"
            ),
            name="Bi-LSTM"
        )
        self.average_pooling = tf.keras.layers.GlobalAveragePooling1D(name="Average")
        self.output_layer = tf.keras.layers.Dense(
            n_tags,
            activation="softmax",
            name="Output"
        )

    def call(self, inputs, training=False):
        """
        Forward pass:
        inputs: (batch_size, max_length) integer token indices
        outputs: (batch_size, n_tags) probabilities per tag after pooling and Dense
        """
        x = self.embedding(inputs)                # (batch_size, max_length, embedding_size)
        x = self.encoder(x, training=training)   # (batch_size, max_length, hidden_state_encoder_size*2)
        x = self.average_pooling(x)               # (batch_size, hidden_state_encoder_size*2)
        x = self.output_layer(x)                   # (batch_size, n_tags)
        return x

def my_model_function():
    # Since the original issue used variables like n_words, n_tags, max_length,
    # we have to provide example values here for demonstration.
    # In practice, these values should come from dataset preprocessing.
    # Assumptions based on typical values and code context:
    # max_length = 80 (from padding used)
    # n_words and n_tags must be provided realistically.
    n_words = 5000  # let's assume vocabulary size ~5000 words
    n_tags = 10     # assume 10 different tags for sequence chunking
    max_length = 80
    embedding_size = 80
    hidden_state_encoder_size = 100
    dropout_rate = 0.5

    return MyModel(n_words, n_tags, max_length, embedding_size, hidden_state_encoder_size, dropout_rate)

def GetInput():
    # Return a random tensor matching the input shape expected by MyModel
    # Input shape: (batch_size, max_length)
    # Values are integer indices in [0, n_words]
    # We must match the assumptions from my_model_function()
    batch_size = 64
    max_length = 80
    n_words = 5000

    # Random integer indices for word tokens in range [1, n_words]
    # assuming 0 might be padding index, so use 1-based index for real words
    input_tensor = tf.random.uniform(
        shape=(batch_size, max_length),
        minval=1,
        maxval=n_words + 1,
        dtype=tf.int32
    )
    return input_tensor

