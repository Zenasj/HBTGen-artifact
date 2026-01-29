# tf.random.uniform((B, 100), dtype=tf.int32) ← Input shape inferred: sequence length 100, integer word indices

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, 
                 n_unique_words=5000,  # assumed vocab size
                 n_dim=64,             # embedding dimension from example
                 max_review_length=100,
                 n_dense=64,
                 dropout=0.5,
                 batch_size=None,      # default None for flexible batch sizes
                 output_size=10,       # for illustrative purposes (e.g. number of classes)
                 dropout_rate=0.5):
        super().__init__()

        # Model A: The embedding + Dense + Dropout model from the original example
        self.embedding = tf.keras.layers.Embedding(
            input_dim=n_unique_words, output_dim=n_dim, 
            input_length=max_review_length, name="embedding"
        )
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(n_dense, activation='relu')
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.dense2 = tf.keras.layers.Dense(1, activation='sigmoid')

        # Model B: The Conv1D + BatchNormalization + LSTM model from the last example
        # Including batch norm differences that caused discrepancy
        # For simplicity set trainable=True, fused=False according to comment
        # Also allow batch_size parameter for LSTM
        input_shape = (max_review_length, 3)  # assuming 3 features per timestep for model B input shape

        self.batchnorm = tf.keras.layers.BatchNormalization(trainable=True, fused=False)
        self.conv1 = tf.keras.layers.Conv1D(10, kernel_size=3, padding='same', activation=None)
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.conv2 = tf.keras.layers.Conv1D(20, kernel_size=2, padding='same', activation=None)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
        self.lstm1 = tf.keras.layers.LSTM(128, return_sequences=True, batch_size=batch_size)
        self.dropout3 = tf.keras.layers.Dropout(dropout_rate)
        self.lstm2 = tf.keras.layers.LSTM(128, batch_size=batch_size)
        self.dense_output = tf.keras.layers.Dense(output_size, activation='softmax')

    def call(self, inputs, training=False):
        """
        inputs: tuple (x1, x2)
          x1: for model A: (batch, sequence_length) integer word indices for embedding
          x2: for model B: (batch, sequence_length=100, features=3) float input for Conv1D+LSTM model

        Returns:
          Boolean tensor indicating if outputs from both models are close numerically.
          Also returns numeric difference as a side output for inspection.
        """

        x1, x2 = inputs

        # Forward pass Model A
        emb = self.embedding(x1)
        flat = self.flatten(emb)
        d1 = self.dense1(flat)
        d1d = self.dropout(d1, training=training)
        out_a = self.dense2(d1d)  # shape (batch, 1)

        # Forward pass Model B
        bn = self.batchnorm(x2, training=training)
        c1 = self.conv1(bn)
        d1_b = self.dropout1(c1, training=training)
        c2 = self.conv2(d1_b)
        d2_b = self.dropout2(c2, training=training)
        lstm1_out = self.lstm1(d2_b, training=training)
        d3_b = self.dropout3(lstm1_out, training=training)
        lstm2_out = self.lstm2(d3_b, training=training)
        out_b = self.dense_output(lstm2_out)  # shape (batch, output_size)

        # For comparison, we reduce dimensionality of Model B output to match Model A output
        # Let's just take first output neuron for simplicity
        out_b_reduced = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x[:, 0], axis=-1))(out_b)

        # Compute difference and comparison
        diff = tf.abs(out_a - out_b_reduced)
        # Tolerance threshold for closeness (example epsilon)
        epsilon = 1e-5
        are_close = tf.reduce_all(diff < epsilon, axis=-1)  # shape (batch,)

        # Return both boolean comparison and diff for detailed inspection
        return are_close, diff

def my_model_function():
    """
    Returns an instance of MyModel with suggested default parameters.
    """
    return MyModel()

def GetInput():
    """
    Generate inputs compatible with MyModel call, i.e., tuple (x1, x2):
        x1: integer tensor (batch, 100) with random word indices in vocab size range
        x2: float tensor (batch, 100, 3) for Conv1D + LSTM model
    
    Assumptions:
    - batch size 4
    - n_unique_words=5000
    - sequence length=100
    - features=3 for model B input
    """
    batch_size = 4
    sequence_length = 100
    n_unique_words = 5000

    x1 = tf.random.uniform(
        shape=(batch_size, sequence_length),
        minval=0, maxval=n_unique_words,
        dtype=tf.int32
    )
    x2 = tf.random.uniform(
        shape=(batch_size, sequence_length, 3),
        minval=-1.0, maxval=1.0,
        dtype=tf.float32
    )
    return (x1, x2)

