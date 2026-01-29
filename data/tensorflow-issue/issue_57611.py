# tf.random.uniform((1, 32, 1), dtype=tf.float32) ← inferred input shape from original issue's example

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Conv1D layer with 1 filter, kernel size 1 as per the issue repro code
        self.conv = tf.keras.layers.Conv1D(filters=1, kernel_size=1)
        # Following layer is ReLU activation which triggers the XLA cudnn bug in TF 2.9.1 GPU image
        self.relu = tf.keras.layers.ReLU()
    
    def call(self, inputs):
        x = self.conv(inputs)
        x = self.relu(x)
        return x

def my_model_function():
    # Return an instance of the model, no extra weights needed beyond initialization
    return MyModel()

def GetInput():
    # Return a random float32 tensor with shape matching expected input: (batch=1, length=32, channels=1)
    # Using uniform distribution over [0, 1)
    return tf.random.uniform((1, 32, 1), dtype=tf.float32)

