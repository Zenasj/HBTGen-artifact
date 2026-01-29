# tf.random.uniform((B, 1, 49), dtype=tf.float32) ← Input shape inferred from create_gru_model

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # GRU with 2 units, input shape: (1, 49) time steps = 1, features = 49
        # Matches the example in the issue with tf.keras.layers.GRU(2)
        self.gru = tf.keras.layers.GRU(2, name='gru')
        # Dense output layer with 49 units and softmax activation
        self.dense = tf.keras.layers.Dense(49, activation=tf.nn.softmax, name='output')

    def call(self, inputs):
        x = self.gru(inputs)
        # No explicit flatten needed as GRU outputs (batch_size, 2)
        # Dense layer maps (batch_size, 2) → (batch_size, 49)
        out = self.dense(x)
        return out

def my_model_function():
    # Initialize the model. Compilation or loading weights if needed can be done here.
    return MyModel()

def GetInput():
    # Generate a random input tensor matching the shape expected by the model:
    # batch_size = 1 (arbitrary), time steps = 1, features = 49
    # dtype float32 to match model input expectations.
    return tf.random.uniform(shape=(1, 1, 49), dtype=tf.float32)

