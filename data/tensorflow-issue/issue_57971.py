# tf.random.uniform((1, 1), dtype=tf.float32) ← Inferred input shape from the original Keras model: input_shape=[1]

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # A simple Sequential model matching the example in the issue
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(units=1, input_shape=[1]),
            tf.keras.layers.Dense(units=16, activation='relu'),
            tf.keras.layers.Dense(units=1)
        ])

    def call(self, x):
        return self.model(x)


def my_model_function():
    # Return an instance of MyModel
    return MyModel()


def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    # Shape: (batch=1, features=1), dtype float32 is the common dtype here
    return tf.random.uniform((1, 1), dtype=tf.float32)

