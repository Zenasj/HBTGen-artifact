# tf.random.uniform((B, 100), dtype=tf.float32) ← The input shape inferred from the example is (batch_size, 100) features

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Based on the original example Dense(10, activation='sigmoid') then Dense(1, activation='sigmoid')
        self.dense1 = tf.keras.layers.Dense(10, activation='sigmoid', name="Hidden_1")
        self.dense2 = tf.keras.layers.Dense(1, activation='sigmoid', name="Output")

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return x

def my_model_function():
    # Return an instance of MyModel initialized freshly
    return MyModel()

def GetInput():
    # Return a random float32 tensor of shape (batch_size, 100)
    # Batch size can be a reasonable default like 40 to match the example
    return tf.random.uniform((40, 100), dtype=tf.float32)

