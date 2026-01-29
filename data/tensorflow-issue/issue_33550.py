# tf.random.uniform((2, 1), dtype=tf.float32) ← inferred input shape from features np.array([[-1.], [1.]], dtype=np.float32)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # A simple single Dense layer with sigmoid activation, as per the example
        self.dense = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs, training=False):
        return self.dense(inputs)

def my_model_function():
    # Return an instance of MyModel, no special initialization or pretrained weights needed
    return MyModel()

def GetInput():
    # Return a random tensor input consistent with the example input features shape (batch 2, feature 1)
    # Use float32 as per example dtype
    return tf.random.uniform((2, 1), dtype=tf.float32)

