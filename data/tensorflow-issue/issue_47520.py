# tf.random.uniform((B, 2), dtype=tf.float32) ← Input shape inferred from example x = [[0,0],[0,1],[1,0],[1,1]]

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define a single Dense layer with 1 unit and sigmoid activation as in the example
        self.dense = tf.keras.layers.Dense(units=1, activation='sigmoid')
    
    def call(self, inputs):
        return self.dense(inputs)

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random tensor matching the input (batch size 4, 2 features)
    # Using batch size 4 as in the example provided
    batch_size = 4
    input_shape = (batch_size, 2)
    # Uniform values between 0 and 1 to match the example inputs of 0s and 1s
    return tf.random.uniform(input_shape, minval=0, maxval=1, dtype=tf.float32)

