# tf.random.uniform((32, 100), dtype=tf.int32)  ← Assumed input shape: batch_size=32, sequence_length=100 (embedding dim used as sequence length in original)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, vocab_size=300, embedding_dim=100, rnn_units=256, batch_size=32):
        super().__init__()
        # Note: In original code batch_input_shape for Embedding uses embedding_dim as sequence length,
        # which is atypical but copied here to reflect original example.
        # Usually sequence length would be >1 and embedding_dim is the embedding vector size.
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.rnn_units = rnn_units

        # Embedding layer with input shape as (batch_size, embedding_dim) int indices
        self.embedding = tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            batch_input_shape=[batch_size, embedding_dim],
            embeddings_initializer='uniform',
            mask_zero=False,  # not specified in original
            trainable=True
        )

        self.lstm = tf.keras.layers.LSTM(
            rnn_units,
            return_sequences=True,
            stateful=False,
            recurrent_activation='sigmoid',
            recurrent_initializer='glorot_uniform',
        )

        self.dense = tf.keras.layers.Dense(vocab_size)

    def call(self, inputs, training=False):
        # inputs: int tensor shape (batch_size, embedding_dim) where embedding_dim used as sequence length
        x = self.embedding(inputs)  # (batch_size, embedding_dim, embedding_dim)
        x = self.lstm(x)            # (batch_size, embedding_dim, rnn_units)
        x = self.dense(x)           # (batch_size, embedding_dim, vocab_size)
        return x

def my_model_function():
    # Return an instance of MyModel with default parameters matching the issue example
    model = MyModel()
    # As in original, the model should be compiled for usage, though not required for inference:
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    return model

def GetInput():
    # Construct a random tensor of int indices representing batch input to Embedding
    # original shape used embedding_dim=100 as sequence length, batch_size=32
    batch_size = 32
    sequence_length = 100  # from embedding_dim in original batch_input_shape
    vocab_size = 300

    # Input shape: (batch_size=32, sequence_length=100)
    # Values: random integers in [0, vocab_size)
    input_tensor = tf.random.uniform(
        shape=(batch_size, sequence_length),
        minval=0,
        maxval=vocab_size,
        dtype=tf.int32
    )
    return input_tensor

