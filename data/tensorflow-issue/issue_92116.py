# tf.random.uniform((B, 20), dtype=tf.float32)  # Inferred input shape: batch size B, feature dimension 20

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Simple feedforward network matching the example:
        # Input shape: (20,)
        # Architecture: Dense(64, relu) -> Dense(10)
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(10)
        
    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return x

def my_model_function():
    # Return an instance of MyModel with default initialization
    return MyModel()

def GetInput():
    # Return a random uniform tensor shaped (batch_size, 20) matching input_shape=(20,)
    # Use batch size 32 as a common default for testing
    return tf.random.uniform((32, 20), dtype=tf.float32)

