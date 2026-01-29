# tf.random.uniform((B=64, H=50, W=100), dtype=tf.float32) ← inferred input shape: batch size 64, sequence length 50, features 100

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, n_classes=10, n_features=100, seq_length=50):
        super().__init__()
        # Bidirectional LSTM with 128 units, no return sequences (last step output)
        self.bidirectional_lstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(128, return_sequences=False)
        )
        # Dense layer with softmax activation for classification
        self.dense = tf.keras.layers.Dense(n_classes, activation="softmax")
    
    def call(self, inputs, training=False):
        # inputs expected shape: (batch_size, seq_length, n_features)
        x = self.bidirectional_lstm(inputs)
        x = self.dense(x)
        return x

def my_model_function():
    # Return an instance of MyModel; model is untrained here
    return MyModel()

def GetInput():
    # Return a random tensor input matching the expected input shape
    # Note: batch size inferred from typical usage in original code's fit (64).
    # However, to be flexible, a default batch size 64 is used.
    batch_size = 64
    seq_length = 50
    n_features = 100
    return tf.random.uniform((batch_size, seq_length, n_features), dtype=tf.float32)

