# tf.random.uniform((B, 10, 3), dtype=tf.float32) and tf.random.uniform((B, 5, 10), dtype=tf.float32) ← input shapes for options and choices respectively

import tensorflow as tf

class Chooser(tf.keras.layers.Layer):
    @tf.function
    def call(self, options_input, choices_input_logits):
        # Apply softmax over the last dimension of choices_input_logits (axis=2)
        choices = tf.nn.softmax(choices_input_logits, axis=2)
        # Perform a batched matrix multiplication: (B,5,10) x (B,10,3) -> (B,5,3)
        result = tf.linalg.matmul(choices, options_input)
        return result

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.chooser = Chooser()
        # Flatten and then Dense(1)
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(1)

    @tf.function(jit_compile=True)
    def call(self, inputs):
        # inputs is a tuple or list: (options_input, choices_input)
        options_input, choices_input = inputs
        net = self.chooser(options_input, choices_input)
        net = self.flatten(net)
        net = self.dense(net)
        return net

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Generate a random input matching the expected shapes:
    # options: (batch_size, 10, 3) uniform in [-1,1]
    # choices: (batch_size, 5, 10) uniform in [0,1]
    batch_size = 32  # batch size consistent with the example
    options = tf.random.uniform(shape=(batch_size, 10, 3), minval=-1.0, maxval=1.0, dtype=tf.float32)
    choices = tf.random.uniform(shape=(batch_size, 5, 10), minval=0.0, maxval=1.0, dtype=tf.float32)
    return (options, choices)

