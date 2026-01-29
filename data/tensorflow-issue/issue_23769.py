# tf.random.uniform((32, 20), dtype=tf.float32) ← inferred input shape from the minimal example (batch=32, features=20)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define the layers matching original model from issue description
        self.dense1 = tf.keras.layers.Dense(10, activation='relu')
        self.dense2 = tf.keras.layers.Dense(5)
    
    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        return self.dense2(x)

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random tensor input matching the input expected by MyModel:
    # batch size 32 (from the original example)
    return tf.random.uniform((32, 20), dtype=tf.float32)

