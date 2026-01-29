# tf.random.uniform((B, 1), dtype=tf.float32) ← Based on input shape of model Input((1,))

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Single Dense layer as per the example model
        self.dense = tf.keras.layers.Dense(1)

    def call(self, inputs, training=False):
        return self.dense(inputs)


def my_model_function():
    # Return an instance of MyModel with initialized weights
    return MyModel()


def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    # Assuming a batch size of 4 for demonstration, 1 feature per input as per example
    return tf.random.uniform((4, 1), dtype=tf.float32)

