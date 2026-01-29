# tf.random.uniform((BATCH_SIZE, MAX_LEN), dtype=tf.int32) ← inferred input shape from padded text sequences with maxlen=172

import tensorflow as tf
from tensorflow.keras import layers

# To align with the original Keras Functional model, we define a subclassed model.
# The original input shape was (MAX_LEN,) i.e. a sequence of token IDs, padded to length 172.
# Embedding input_dim=500, output_dim=62, followed by Bidirectional LSTM(32),
# Dense(64,relu), Dense(32,relu), Dense(1) (regression output).

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.embedding = layers.Embedding(input_dim=500, output_dim=62)
        self.bi_lstm = layers.Bidirectional(layers.LSTM(32))
        self.dense1 = layers.Dense(64, activation='relu')
        self.dense2 = layers.Dense(32, activation='relu')
        self.out = layers.Dense(1)  # Regression output

    def call(self, inputs, training=False):
        x = self.embedding(inputs)  # (B, MAX_LEN, 62)
        x = self.bi_lstm(x)         # (B, 64) since lstm units=32 per direction
        x = self.dense1(x)          # (B, 64)
        x = self.dense2(x)          # (B, 32)
        return self.out(x)          # (B, 1)

def my_model_function():
    # Instantiate the model
    model = MyModel()
    # Compile similar to original code: MSE loss, Adam optimizer, accuracy metric (though accuracy is unusual for regression)
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    return model

def GetInput():
    # Return a random tensor of shape (BATCH_SIZE, MAX_LEN)
    # Tokens are integer indices from 0 to 499 (since num_words=500 in original tokenizer)
    BATCH_SIZE = 16
    MAX_LEN = 172
    # Random integers simulating tokenized, padded sequences
    input_tensor = tf.random.uniform(shape=(BATCH_SIZE, MAX_LEN), minval=0, maxval=500, dtype=tf.int32)
    return input_tensor

