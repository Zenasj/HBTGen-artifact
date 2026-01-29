# tf.random.uniform((B, 1), dtype=tf.int32), tf.random.uniform((B, 128, 3), dtype=tf.float32)  ← inferred input shapes for inputs

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Embedding branch
        self.embedding = layers.Embedding(1000, 3)
        self.flatten = layers.Flatten()
        self.dense_embed_1 = layers.Dense(256, activation="relu")
        self.dense_embed_2 = layers.Dense(128, activation="relu")
        self.dense_embed_3 = layers.Dense(64, activation="relu")

        # LSTM branch
        self.bilstm1 = layers.Bidirectional(layers.LSTM(128, activation='relu', return_sequences=True))
        self.bilstm2 = layers.Bidirectional(layers.LSTM(64, activation='relu'))

        # Combined fully connected layers for output z
        self.dense_combined_1 = layers.Dense(2, activation="relu")
        self.dense_combined_2 = layers.Dense(1, activation="linear")

        # Combined fully connected layers for output t
        self.dense_combined_t1 = layers.Dense(2, activation="relu")
        self.dense_combined_t2 = layers.Dense(1, activation="linear")

    def call(self, inputs, training=False):
        inputA, inputB = inputs  # inputA shape: (B,1), inputB shape: (B,128,3)

        # Embedding branch forward
        x = self.embedding(inputA)  # (B, 1, 3)
        x = self.flatten(x)         # (B, 3)
        x = self.dense_embed_1(x)
        x = self.dense_embed_2(x)
        x = self.dense_embed_3(x)   # (B, 64)

        # LSTM branch forward
        y = self.bilstm1(inputB)    # (B, 128, 256)
        y = self.bilstm2(y)         # (B, 128)

        # Concatenate embedding and LSTM outputs
        combined = tf.concat([x, y], axis=-1)  # (B, 64+128=192)

        # Output z branch
        z = self.dense_combined_1(combined)
        z = self.dense_combined_2(z)  # (B, 1)

        # Output t branch
        t = self.dense_combined_t1(combined)
        t = self.dense_combined_t2(t)  # (B, 1)

        return [z, t]

def my_model_function():
    """
    Returns an instance of MyModel,
    Compiled with RMSprop optimizer without momentum (to avoid known embedding+momentum bugs in TF 1.14).
    """
    model = MyModel()
    # We compile model here to match the original reported usage
    # Use RMSprop with learning rate and decay; momentum disabled per issue workaround
    optimizer = RMSprop(learning_rate=0.001, decay=0.05, momentum=0.0)
    model.compile(optimizer=optimizer, loss=['mse', 'mse'])
    return model

def GetInput(batch_size=128):
    """
    Returns a tuple of inputs matching MyModel's expected inputs.
    inputA: integer indices for embedding: shape (batch_size, 1), values in [0, 999]
    inputB: float data for LSTM: shape (batch_size, 128, 3)
    """
    inputA = tf.random.uniform((batch_size, 1), minval=0, maxval=1000, dtype=tf.int32)
    inputB = tf.random.uniform((batch_size, 128, 3), dtype=tf.float32)
    return (inputA, inputB)

