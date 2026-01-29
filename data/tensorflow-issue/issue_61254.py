# tf.random.uniform((B, 10), dtype=tf.int32) ← Input shape inferred from Input(shape=10) in the provided GRU model example

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, num_units=25, input_shape=10):
        super().__init__()
        # Embedding layer with input_dim=100, output_dim=10 as in the example
        self.embedding = tf.keras.layers.Embedding(input_dim=100, output_dim=10, input_length=input_shape)
        # GRU layer with 32 units, dropout=0.7338, return_sequences=True as per the example
        self.gru = tf.keras.layers.GRU(units=32, dropout=0.7338014982069313, return_sequences=True)
        # ActivityRegularization layer with l1 and l2 negative factors as per issue reproduction
        # Note: Although docs say l1 and l2 should be positive, actual implementation accepts negatives.
        self.activity_reg = tf.keras.layers.ActivityRegularization(l1=-0.616784030867379, l2=-0.9646777799675004)
        # Dense output layer with units=25 and relu activation
        self.dense = tf.keras.layers.Dense(units=num_units, activation="relu")
        # Flatten layer to flatten the output to shape (batch_size, 250)
        self.flatten = tf.keras.layers.Flatten()

    def call(self, inputs, training=None):
        x = self.embedding(inputs)  # shape: (B, 10, 10)
        x = self.gru(x, training=training)  # shape: (B, 10, 32)
        x = self.activity_reg(x)  # shape: (B, 10, 32), no params added, just regularization
        x = self.dense(x)  # shape: (B, 10, 25)
        x = self.flatten(x)  # shape: (B, 250)
        return x

def my_model_function():
    # Return an instance of MyModel with default parameters matching the original example
    return MyModel()

def GetInput():
    # Return a random integer tensor of shape (batch_size=4, sequence_length=10), values in [0, 99]
    # Matches the input expected by Embedding layer (input_dim=100)
    return tf.random.uniform(shape=(4, 10), minval=0, maxval=100, dtype=tf.int32)

