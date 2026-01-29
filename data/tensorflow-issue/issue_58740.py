# tf.random.uniform((B, 3), dtype=tf.float32) ← Inferred input shape: batch size variable, features=3

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Simple two-layer MLP matching the one from the issue snippet
        self.dense1 = tf.keras.layers.Dense(4, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(5, activation=tf.nn.softmax)
    
    def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)


def my_model_function():
    # Return an instance of MyModel
    return MyModel()


def GetInput():
    # Return a random input tensor matching expected input shape (batch_size, 3)
    # Using batch size = 2 as a reasonable example. dtype float32 as Keras default.
    return tf.random.uniform((2, 3), dtype=tf.float32)

