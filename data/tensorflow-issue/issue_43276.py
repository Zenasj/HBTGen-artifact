# tf.random.uniform((1, None), dtype=tf.int32) ← Input shape: batch size 1, variable sequence length (None), integer token IDs

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, mid_units, batch_size):
        super().__init__()
        # Embedding layer with fixed batch input shape [batch_size, None]
        self.embedding = tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            batch_input_shape=[batch_size, None]
        )
        # Following dense layers with activations and dropouts per original model architecture
        self.dense1 = tf.keras.layers.Dense(2500, activation='relu')
        self.dropout1 = tf.keras.layers.Dropout(0.15)
        self.dense2 = tf.keras.layers.Dense(3500, activation='relu')
        self.dense3 = tf.keras.layers.Dense(5500, activation='relu')
        self.dropout2 = tf.keras.layers.Dropout(0.15)

        self.dense4 = tf.keras.layers.Dense(7500, activation='relu')
        self.dense5 = tf.keras.layers.Dense(9500, activation='relu')
        self.batchnorm1 = tf.keras.layers.BatchNormalization()
        self.dropout3 = tf.keras.layers.Dropout(0.15)

        self.dense6 = tf.keras.layers.Dense(mid_units, activation='relu')
        self.dense7 = tf.keras.layers.Dense(mid_units, activation='relu')
        self.dropout4 = tf.keras.layers.Dropout(0.15)

        self.dense8 = tf.keras.layers.Dense(1500, activation='relu')
        self.dense9 = tf.keras.layers.Dense(500, activation='relu')
        self.dropout5 = tf.keras.layers.Dropout(0.15)

        # Final activation and normalization
        self.activation = tf.keras.layers.Activation('softmax')
        self.batchnorm2 = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training=False):
        """
        inputs: Tensor of shape [batch_size, sequence_length] with integer token IDs
        """
        x = self.embedding(inputs)                    # Shape: (batch_size, seq_len, embedding_dim)
        x = self.dense1(x)
        if training:
            x = self.dropout1(x, training=training)
        x = self.dense2(x)
        x = self.dense3(x)
        if training:
            x = self.dropout2(x, training=training)
        x = self.dense4(x)
        x = self.dense5(x)
        x = self.batchnorm1(x, training=training)
        if training:
            x = self.dropout3(x, training=training)
        x = self.dense6(x)
        x = self.dense7(x)
        if training:
            x = self.dropout4(x, training=training)
        x = self.dense8(x)
        x = self.dense9(x)
        if training:
            x = self.dropout5(x, training=training)
        x = self.activation(x)
        x = self.batchnorm2(x, training=training)
        return x

def my_model_function():
    # Assumptions for parameters based on the original issue:
    # vocab_size inferred from the vocab variable length or user needs to provide
    # embedding_dim was unclear but model summary shows embedding output dim=8800,
    # likely vocab_size=some large number, embedding_dim=8800 (per embedding_4 output shape)
    # mid_units=7000 per model.build argument in issue
    # batch_size=1 as per model summary shapes
    # To keep the model reasonable, assume:
    vocab_size = 20000  # Placeholder; user should replace with actual vocab size
    embedding_dim = 8800
    mid_units = 7000
    batch_size = 1
    return MyModel(vocab_size=vocab_size, embedding_dim=embedding_dim, mid_units=mid_units, batch_size=batch_size)

def GetInput():
    # Generate a random batch of token IDs
    # batch_size=1, sequence length variable but some fixed length needed for testing, e.g. 10
    batch_size = 1
    seq_length = 10
    vocab_size = 20000  # Must match model vocab size

    # Create random int tensor [1, 10] in [0, vocab_size)
    return tf.random.uniform(
        shape=(batch_size, seq_length),
        minval=0,
        maxval=vocab_size,
        dtype=tf.int32
    )

