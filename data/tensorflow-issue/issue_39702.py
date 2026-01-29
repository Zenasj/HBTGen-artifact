# tf.random.uniform((32, 28, 28), dtype=tf.float32)
import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras import Model

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.flatten = Flatten()
        self.dense1 = Dense(64, activation='relu')
        self.dense2 = Dense(10)

    def call(self, inputs, training=False):
        # inputs is expected as a single tensor: data (batch_size, 28, 28)
        x = self.flatten(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        return x


def my_model_function():
    # Return an instance of MyModel
    return MyModel()


def GetInput():
    # Return a random tensor input that matches the input expected by MyModel:
    # Shape inferred from example code: (batch_size=32, 28, 28), dtype float32
    return tf.random.uniform((32, 28, 28), dtype=tf.float32)

