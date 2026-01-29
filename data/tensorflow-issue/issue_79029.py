# tf.random.uniform((1,), dtype=tf.float32) ← input shape inferred from issue input_shape=(1,)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define a simple single Dense layer model matching the original Sequential example
        self.dense = tf.keras.layers.Dense(1)

    def call(self, inputs):
        return self.dense(inputs)

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random tensor input matching input shape (1,) with dtype float32
    return tf.random.uniform((1,), dtype=tf.float32)

