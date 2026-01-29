# tf.random.uniform((B, 1), dtype=tf.float32) ← Input shape inferred as batch of scalars simulating original demo input shape (e.g., (4,1))

import tensorflow as tf

class MyModel(tf.keras.Model):
    """
    A simple linear model with one Dense layer (5 units), replicating the original 'Net' class.
    Supports restoration of kernel and bias variables in a delayed manner using tf.train.Checkpoint.
    """
    def __init__(self):
        super(MyModel, self).__init__()
        self.l1 = tf.keras.layers.Dense(5)

    def call(self, x):
        return self.l1(x)

def my_model_function():
    # Return an instance of MyModel, weights are uninitialized until run on input.
    return MyModel()

def GetInput():
    # Return a random tensor input shaped (batch_size, 1) matching the model input size.
    # The original example used input shape (4,1) but batch size can be flexible.
    # Using batch size 4 for consistency.
    return tf.random.uniform((4, 1), dtype=tf.float32)

