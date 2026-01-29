# tf.random.uniform((B, 16, 5), dtype=tf.float32) ← batch size B is variable, height=16, width=5 input shape for dense layer

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Dense layer with 20 units and relu activation, expects input shape (16, 5)
        self.dense = tf.keras.layers.Dense(20, activation='relu')

    def call(self, inputs, *args, **kwargs):
        x = self.dense(inputs)
        return x


def my_model_function():
    # Create and return an instance of MyModel
    return MyModel()


def GetInput():
    # Generate a random input tensor matching the expected input shape (batch_size, 16, 5)
    # batch size chosen as 8 for example
    batch_size = 8
    return tf.random.uniform((batch_size, 16, 5), dtype=tf.float32)

