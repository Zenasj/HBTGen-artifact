# tf.random.uniform((B, 10), dtype=tf.float32) ← inferred input shape based on model input_shape=[10]

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Simple Sequential model with one Dense layer as in provided example
        self.dense = tf.keras.layers.Dense(1, input_shape=[10])

    def call(self, inputs, training=False):
        # Forward pass through dense layer
        return self.dense(inputs)

def my_model_function():
    # Return an initialized instance of MyModel
    return MyModel()

def GetInput():
    # Return a random tensor matching the input shape expected by MyModel
    # Shape: (batch_size=4, features=10), dtype float32
    # batch size chosen arbitrary to allow demonstration compatible with the model
    return tf.random.uniform((4, 10), dtype=tf.float32)

