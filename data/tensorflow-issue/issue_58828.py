# tf.random.uniform((B, 128), dtype=tf.float32) ← Input shape inferred from the provided sample (2048, 128)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Based on the simple 3-layer fully-connected network described in the issue
        self.dense1 = tf.keras.layers.Dense(128, activation='tanh')
        self.dense2 = tf.keras.layers.Dense(128)
        
    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return x

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # According to the reproduced code, input shape is (batch_size, 128)
    # Use batch_size=2048 as per the typical training input used
    return tf.random.uniform((2048, 128), dtype=tf.float32)

