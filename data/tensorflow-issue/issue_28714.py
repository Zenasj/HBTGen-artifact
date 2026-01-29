# tf.random.uniform((64, 784), dtype=tf.float32) ← batch size 64, image flattened 28*28=784
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Equivalent to the example model:
        # Input shape: (batch, 784)
        # Architecture: Dense(64, relu) -> Dense(64, relu) -> Dense(10, softmax)
        self.dense_1 = layers.Dense(64, activation='relu', name='dense_1')
        self.dense_2 = layers.Dense(64, activation='relu', name='dense_2')
        self.predictions = layers.Dense(10, activation='softmax', name='predictions')

    def call(self, inputs, training=False):
        x = self.dense_1(inputs)
        x = self.dense_2(x)
        output = self.predictions(x)
        return output

def my_model_function():
    # Return an instance of MyModel, uncompiled by default (user can compile if needed)
    return MyModel()

def GetInput():
    # Return a random tensor matching the expected input shape:
    # Batch size 64, flattened MNIST image 784
    return tf.random.uniform((64, 784), dtype=tf.float32)

