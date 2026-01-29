# tf.random.uniform((1,), dtype=tf.float32) ← input shape inferred from the issue example model

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # The model in the issue is a simple 3-layer MLP:
        # Input (shape=(1,))
        # Dense(16) + ReLU
        # Dense(16) + ReLU
        # Dense(1)
        self.dense1 = tf.keras.layers.Dense(16)
        self.relu1 = tf.keras.layers.ReLU()
        self.dense2 = tf.keras.layers.Dense(16)
        self.relu2 = tf.keras.layers.ReLU()
        self.dense3 = tf.keras.layers.Dense(1)
    
    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.relu1(x)
        x = self.dense2(x)
        x = self.relu2(x)
        x = self.dense3(x)
        return x

def my_model_function():
    # Returns an instance of the model
    return MyModel()

def GetInput():
    # Returns a random float32 tensor with shape (1,), matching model input
    # As in the provided code, input data range is uniform between 0 and 2*pi.
    return tf.random.uniform(shape=(1,), minval=0, maxval=2*3.141592653589793, dtype=tf.float32)

