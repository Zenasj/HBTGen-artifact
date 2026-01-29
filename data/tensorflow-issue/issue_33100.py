# tf.random.uniform((32, 4), dtype=tf.float32) ← Based on example input shape (batch=32, features=4)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Simple dense layer with sigmoid activation as per example
        self.dense = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs):
        return self.dense(inputs)

def my_model_function():
    return MyModel()

def GetInput():
    # Return random input matching the example shape (batch of 32, 4 features)
    return tf.random.uniform((32, 4), dtype=tf.float32)

