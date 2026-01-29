# tf.random.uniform((10, 4), dtype=tf.float32) ← Input is 2D tensor with shape (batch_size=10, features=4)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define a simple Dense layer with 1 output and 'softmax' activation
        # Matching the original reported model in the issue: input shape (4,), output 1 with softmax
        self.dense = tf.keras.layers.Dense(1, activation='softmax')

    def call(self, inputs, training=False):
        return self.dense(inputs)

def my_model_function():
    # Instantiate the model and return it
    # No compilation is necessary - matching the reported issue context (compile=False)
    return MyModel()

def GetInput():
    # Return a random tensor input with shape (10, 4) matching the model's input shape
    # Using float32 dtype to align with typical TF defaults
    return tf.random.uniform((10, 4), dtype=tf.float32)

