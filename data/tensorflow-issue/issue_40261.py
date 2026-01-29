# tf.random.uniform((B, 15, 60), dtype=tf.float32) ← input shape inferred from GRU example with shape=(15, 60)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define GRU layer and Dense output layer as per the discussed example
        # dropout and recurrent_dropout are known to cause overhead in TF 2.2+
        # We include them here to reflect the example and discuss in comments
        self.gru = tf.keras.layers.GRU(50, dropout=0.5, recurrent_dropout=0.5)
        self.dense = tf.keras.layers.Dense(1)
    
    def call(self, inputs, training=False):
        # Forward pass: input -> GRU -> Dense
        # Use `training` argument to match dropouts behavior
        x = self.gru(inputs, training=training)
        x = self.dense(x)
        return x


def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random tensor input matching (batch_size=1, 15, 60)
    # Use tf.float32 dtype consistent with typical model input
    # batch size 1 assumed for latency-sensitive usage scenario discussed
    return tf.random.uniform((1, 15, 60), dtype=tf.float32)

