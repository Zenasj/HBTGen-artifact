# tf.random.uniform((B, 300, 6), dtype=tf.float32) ← Input shape inferred from build_model() Input layer = (300, 6)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # LSTM layer with 50 units, dropout & recurrent dropout 0.1, no return sequences (output last state)
        self.lstm = tf.keras.layers.LSTM(
            50,
            dropout=0.1,
            recurrent_dropout=0.1,
            return_sequences=False,
            name='lstm',
        )
        # Dense output layer with 1 unit (regression output)
        self.dense = tf.keras.layers.Dense(1)

    def call(self, inputs, training=False):
        x = inputs
        x = self.lstm(x, training=training)
        x = self.dense(x)
        return x

def my_model_function():
    return MyModel()

def GetInput():
    # Batch size chosen as 4096 (based on original code's batch_size parameter)
    # Input shape: (batch_size, 300, 6)
    batch_size = 4096
    return tf.random.uniform((batch_size, 300, 6), dtype=tf.float32)

