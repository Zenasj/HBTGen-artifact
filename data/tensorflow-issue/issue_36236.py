# tf.random.uniform((B, T, 1), dtype=tf.float32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # GRU layer with 10 units
        self.gru = tf.keras.layers.GRU(10)
        # Dense output layer with 1 unit
        self.dense = tf.keras.layers.Dense(1)

    def call(self, inputs, training=False):
        # Run the GRU on inputs: shape (batch, time, 1) -> (batch, units)
        hidden = self.gru(inputs)
        # Workaround for gradient issue: multiply by 1 to force conversion of IndexedSlices to Tensor
        hidden = tf.gather(hidden * 1, [0])
        output = self.dense(hidden)
        return output

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random input tensor matching shape (batch=1, time_steps=3, features=1)
    # Matches example input from the original issue
    return tf.random.uniform((1, 3, 1), dtype=tf.float32)

