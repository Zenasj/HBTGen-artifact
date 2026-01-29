# tf.random.uniform((B, 1), dtype=tf.float32) - input shape inferred from train_data shape (10,1) in issue example

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense = tf.keras.layers.Dense(1, input_shape=(1,))
    
    def call(self, inputs):
        # Forward pass through dense layer
        return self.dense(inputs)

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Generate a random tensor input matching the expected input shape: (B, 1)
    # Batch size B is set to 10 to match example train_data dimension
    B = 10
    return tf.random.uniform((B, 1), dtype=tf.float32)

