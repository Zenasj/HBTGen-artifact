# tf.random.uniform((B, 1), dtype=tf.float32) ← Input shape inferred from example code: batches of (batch_size, 1)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # a Dense layer with one unit, no bias, kernel init all ones, replicating the example model
        self.dense = tf.keras.layers.Dense(
            1,
            activation=None,
            use_bias=False,
            kernel_initializer=tf.keras.initializers.Ones())
    
    def call(self, inputs):
        # Forward pass: model(x) = x * kernel (all ones) which is identity effect without bias or activation
        return self.dense(inputs)

def my_model_function():
    # Return an instance of MyModel with default initialization
    return MyModel()

def GetInput():
    # Generate a random single batch with shape (batch_size, 1) as input for the model
    # Based on example, typical batch_size is 32, values in range [0, 100)
    batch_size = 32
    # Using float32 tensor with random uniform values as example input
    return tf.random.uniform((batch_size, 1), minval=0, maxval=100, dtype=tf.float32)

