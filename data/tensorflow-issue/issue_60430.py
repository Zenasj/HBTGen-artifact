# tf.random.uniform((10, 4), dtype=tf.float32) ← inferred input shape from repro example

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Mimicking the Sequential model from the issue with two Dense layers:
        self.dense1 = tf.keras.layers.Dense(3, input_shape=(4,))
        self.dense2 = tf.keras.layers.Dense(1)

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return x

def my_model_function():
    # Return an instance of MyModel with no pretrained weights
    return MyModel()

def GetInput():
    # Create a random float tensor matching shape (10, 4), like the original repro
    # This shape corresponds to batch size 10, 4-dimensional input features
    return tf.random.uniform((10, 4), dtype=tf.float32)

