# tf.random.uniform((B, 784), dtype=tf.float32) ← batch dimension and 784 input features as seen in provided example

import tensorflow as tf
from tensorflow.keras import layers

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Build toy model as per example in the issue
        self.dense1 = layers.Dense(64, activation="relu")
        self.dense2 = layers.Dense(64, activation="relu")
        self.out = layers.Dense(10)  # output logits for 10 classes
    
    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.dense2(x)
        logits = self.out(x)
        return logits

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Returns a random input tensor matching the shape expected by MyModel: (batch_size, 784)
    # Batch size is chosen as 32 (as example in code)
    batch_size = 32
    input_dim = 784
    # Using tf.random.uniform to generate random floats between 0 and 1 (like np.random.rand)
    return tf.random.uniform((batch_size, input_dim), dtype=tf.float32)

