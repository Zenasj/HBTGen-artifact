# tf.random.uniform((B, None), dtype=tf.float32)  # Ragged sequences of variable length, float features (1D per timestep)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Following the inference from comments and original issue, to handle ragged sequences with variable length and 1 feature per timestep,
        # the input shape is [None], ragged=True, dtype float32.
        # Since no embedding is used on float data, LSTM input expects shape [batch, time, features=1].
        # We'll add a Reshape layer that expands last dimension as 1 (feature dim) from [batch, time] ragged to [batch, time, 1].
        
        self.expand_dims = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1))
        
        self.lstm1 = tf.keras.layers.LSTM(
            units=64,
            activation='tanh',
            dropout=0.2,
            return_sequences=True,
            name='LSTM_1'
        )
        self.lstm2 = tf.keras.layers.LSTM(
            units=256,
            activation='tanh',
            dropout=0.2,
            return_sequences=True,
            name='LSTM_2'
        )
        self.lstm3 = tf.keras.layers.LSTM(
            units=100,
            activation='tanh',
            dropout=0.2,
            return_sequences=False,
            name='LSTM_3'
        )
        self.dense = tf.keras.layers.Dense(
            units=4,
            activation='sigmoid',
            name='Dense_1'
        )
    
    def call(self, inputs, training=None):
        # inputs is RaggedTensor with shape [batch, None]
        # Expand dims to add feature dimension (1)
        x = self.expand_dims(inputs)
        # LSTM layers expect dense tensor, so RaggedTensor will be automatically converted or requires to_dense.
        # To avoid issues with padding, LSTM in TF 2.x supports ragged input directly.
        x = self.lstm1(x, training=training)
        x = self.lstm2(x, training=training)
        x = self.lstm3(x, training=training)
        x = self.dense(x)
        return x

def my_model_function():
    # Returns an instance of MyModel initialized and compiled with commonly used optimizer, loss and metrics.
    model = MyModel()
    # Compile model with categorical crossentropy and Adam optimizer as per original usage.
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
        metrics=['accuracy']
    )
    return model

def GetInput():
    # Return a random ragged tensor input matching the expected input of MyModel:
    # - ragged shape: [batch_size, None]
    # - dtype float32
    # We generate random ragged tensor sequences with variable lengths between 5 and 20
    import numpy as np
    batch_size = 32
    # Generate a list of 32 sequences with variable length between 5 and 20
    sequences = []
    for _ in range(batch_size):
        seq_len = np.random.randint(5, 21)  # length between 5 and 20
        seq = np.random.uniform(low=0.0, high=1.0, size=(seq_len,)).astype(np.float32)
        sequences.append(seq)
    # Create ragged tensor from sequences
    ragged_input = tf.ragged.constant(sequences, dtype=tf.float32)
    return ragged_input

