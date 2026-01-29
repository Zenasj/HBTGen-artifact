# tf.random.uniform((B, 5), dtype=tf.float32) ← Inputs are two tensors of shape (batch_size, 5)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define two Dense layers explicitly as attributes to avoid saved model loading issues
        self.dense1 = tf.keras.layers.Dense(1, name='dense1')
        self.dense2 = tf.keras.layers.Dense(1, name='dense2')

    def call(self, inputs, training=False):
        # Expect inputs as a tuple/list of two tensors each of shape (batch_size, 5)
        input1, input2 = inputs
        # Pass through respective dense layers
        out1 = self.dense1(input1)
        out2 = self.dense2(input2)
        # Calculate difference
        out = out1 - out2
        # Apply sigmoid activation
        out = tf.nn.sigmoid(out)
        return out

def my_model_function():
    # Return a new instance of MyModel
    # No special initialization needed beyond __init__
    return MyModel()

def GetInput():
    # Generate example inputs matching the expected input shape: two tensors with shape (batch_size, 5)
    # Using batch size = 4 as an example
    batch_size = 4
    input1 = tf.random.uniform((batch_size, 5), dtype=tf.float32)
    input2 = tf.random.uniform((batch_size, 5), dtype=tf.float32)
    return (input1, input2)

