# tf.random.uniform((32, 10), dtype=tf.float32) ← Input shape inferred from numpy array shape (1000,10) batched by 32 for tf.data.Dataset

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Recreating the Sequential model structure described in the issue
        self.dense1 = tf.keras.layers.Dense(32, input_shape=(10,), name="dense_1")
        self.dense2 = tf.keras.layers.Dense(32, name="dense_2")
        self.dense3 = tf.keras.layers.Dense(1, name="dense_3")

    def call(self, inputs, training=None):
        x = self.dense1(inputs)
        x = self.dense2(x)
        output = self.dense3(x)
        return output

def my_model_function():
    # Instantiate the model, no pretrained weights mentioned
    return MyModel()

def GetInput():
    # Return input matching the expected input shape (32, 10) as per batching in dataset in issue
    # Using tf.random.uniform similar to numpy random.randn usage in the original issue example
    return tf.random.uniform((32, 10), dtype=tf.float32)

