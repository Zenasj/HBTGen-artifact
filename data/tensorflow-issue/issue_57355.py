# tf.random.uniform((batch_size, 100, 10), dtype=tf.float32)  ← Inferred input shape from the issue example

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Flatten layer and Dense layer matching the Sequential model from the issue
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(3, activation='softmax')
    
    def call(self, inputs, training=False):
        x = self.flatten(inputs)
        return self.dense(x)

def my_model_function():
    # Returns an instance of MyModel
    return MyModel()

def GetInput():
    # Provide a random input matching the expected input shape to MyModel
    # Shape corresponds to the input defined in the reported issue: (batch_size, 100, 10)
    batch_size = 33  # Chosen batch size similar to example in chunk 2
    return tf.random.uniform((batch_size, 100, 10), dtype=tf.float32)

