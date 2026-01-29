# tf.random.uniform((1, 1, 1), dtype=tf.float32) ← input shape inferred from representative dataset generator in issue

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # GRUCell wrapped in RNN layer with unroll=True, matching original model
        self.rnn = tf.keras.layers.RNN(
            tf.keras.layers.GRUCell(10),
            unroll=True
        )
    
    def call(self, inputs, training=False):
        # Forward pass through GRU RNN layer
        return self.rnn(inputs, training=training)

def my_model_function():
    # Return an instance of MyModel as per the original Keras Sequential model structure
    return MyModel()

def GetInput():
    # Return a random tensor input shaped (batch=1, time=1, features=1) matching original input shape
    # dtype float32 as used in the representative dataset
    return tf.random.uniform((1, 1, 1), dtype=tf.float32)

