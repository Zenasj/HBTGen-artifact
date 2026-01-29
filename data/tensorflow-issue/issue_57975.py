# tf.random.uniform((32, 10, 3), dtype=tf.float32), tf.random.uniform((32, 5, 10), dtype=tf.float32)
import tensorflow as tf

class Chooser(tf.keras.layers.Layer):
    @tf.function
    def call(self, options_input, choices_input_logits):
        # Apply softmax along axis=2 as in original example
        choices = tf.nn.softmax(choices_input_logits, axis=2)
        # Perform batch matrix multiplication: (batch, 5, 10) x (batch, 10, 3) -> (batch, 5, 3)
        result = tf.linalg.matmul(choices, options_input)
        return result

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.chooser = Chooser()
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(1)

    @tf.function
    def call(self, inputs):
        # inputs is tuple/list of two tensors: (options_input, choices_input)
        options_input, choices_input = inputs
        x = self.chooser(options_input, choices_input)
        x = self.flatten(x)
        x = self.dense(x)
        return x

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a tuple with random tensors matching inputs:
    # options_input shape: (batch, 10, 3)
    # choices_input shape: (batch, 5, 10)
    batch_size = 32
    options = tf.random.uniform((batch_size, 10, 3), dtype=tf.float32, minval=-1.0, maxval=1.0)
    choices_logits = tf.random.uniform((batch_size, 5, 10), dtype=tf.float32, minval=0.0, maxval=1.0)
    return (options, choices_logits)

