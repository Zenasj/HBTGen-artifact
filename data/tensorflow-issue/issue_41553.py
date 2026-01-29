# tf.random.uniform((B, 8), dtype=tf.float32)  # Input shape inferred from Dense layer input shape (8,)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Following the example model in the issue: 
        # Input 8 features → Dense(2) → Dense(2)
        self.dense1 = tf.keras.layers.Dense(2)
        self.dense2 = tf.keras.layers.Dense(2)

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return x

def my_model_function():
    # Return an instance of MyModel with default initialization
    return MyModel()

def GetInput():
    # Return a random tensor input compatible with MyModel's expected input shape (batch_size, 8)
    # Batch size assumed 4 as a reasonable default
    batch_size = 4
    input_tensor = tf.random.uniform((batch_size, 8), dtype=tf.float32)
    return input_tensor

