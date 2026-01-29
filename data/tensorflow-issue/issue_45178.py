# tf.random.uniform((batch_size, 784), dtype=tf.float32)
import tensorflow as tf
from tensorflow import keras
import numpy as np

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define a simple 2-layer dense network as in the given code
        self.dense1 = keras.layers.Dense(256, activation="relu")
        self.dense2 = keras.layers.Dense(256, activation="relu")
        self.out = keras.layers.Dense(10)  # logits layer
    
    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.out(x)

def my_model_function():
    # Instantiate the model and compile with loss and optimizer as in the sample
    model = MyModel()
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )
    return model

def GetInput():
    batch_size = 10240  # Taken from the original code batch size
    # Generate dummy random input matching the (batch_size, 784) float32 tensor expected by the model
    # This corresponds to flattened MNIST images normalized (0,1)
    return tf.random.uniform((batch_size, 784), minval=0, maxval=1, dtype=tf.float32)

