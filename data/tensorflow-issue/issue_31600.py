# tf.random.uniform((1, 7), dtype=tf.float32) ← batch size 1, feature dimension 7

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # The model replicates the structure:
        # Dense(7) => Dense(12) => Dense(1)
        # Input shape excludes batch dimension, i.e. (7,)
        self.dense1 = tf.keras.layers.Dense(7, input_shape=(7,))
        self.dense2 = tf.keras.layers.Dense(12)
        self.dense3 = tf.keras.layers.Dense(1)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        output = self.dense3(x)
        return output


def my_model_function():
    # Return an instance of MyModel as defined above
    return MyModel()


def GetInput():
    # Return a batch input tensor (batch size = 1) with shape (1, 7)
    # This matches the expected input shape of the model.
    # Using tf.random.uniform to create a random input sample.
    return tf.random.uniform((1, 7), dtype=tf.float32)

