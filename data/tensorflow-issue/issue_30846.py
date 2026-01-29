# tf.random.uniform((B, 10), dtype=tf.float32) ← inferred input shape from the Keras Input layer shape (10,)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Reshape layer to reshape (B, 10) input to (B, 10, 1)
        self.reshape = tf.keras.layers.Reshape((-1, 1), name='reshape')
        # Dot layer with axes=2 (dot product over last axis of each input)
        self.dot = tf.keras.layers.Dot(axes=2, name='dot')
        # Flatten layer to flatten the output to 2D (B, *)
        self.flatten = tf.keras.layers.Flatten(name='flatten')

    def call(self, inputs):
        # inputs shape: (B, 10)
        x = self.reshape(inputs)          # shape: (B, 10, 1)
        # Dot product of (B, 10, 1) with itself over axes=2 results in (B, 10, 10)
        x = self.dot([x, x])             # shape: (B, 10, 10)
        x = self.flatten(x)              # shape: (B, 100)
        return x

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random tensor input matching expected input: shape (batch_size, 10)
    # Batch size arbitrary, e.g., 2
    return tf.random.uniform((2, 10), dtype=tf.float32)

