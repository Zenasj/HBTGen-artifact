# tf.random.normal((1, 5), dtype=tf.float32) ← inferred input shape from SHAPE=(1, 5)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define a single dense layer with output size 10 as per the original code
        self.dense_layer = tf.keras.layers.Dense(10)
        
    @tf.function(input_signature=[tf.TensorSpec(shape=(1, 5), dtype=tf.float32)])
    def call(self, x):
        # Forward pass through the dense layer
        return self.dense_layer(x)


def my_model_function():
    # Instantiate and return MyModel
    return MyModel()

def GetInput():
    # Generate a random normal tensor with shape (1, 5) and dtype float32 matching the model's input
    return tf.random.normal((1, 5), dtype=tf.float32)

