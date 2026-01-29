# tf.random.uniform((), dtype=tf.float32) ← The model's input is scalar since the issue involves scalar trainable variables

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # A single scalar trainable variable, matching the original example
        self.var = tf.Variable(initial_value=0.1, dtype=tf.float32, trainable=True)

    def call(self, inputs):
        # This model does not actually use the input tensor as in the original example,
        # but to maintain framework compatibility, we accept an input tensor.
        # The output simulates the original loss computation: mean of 10 copies of self.var concatenated.
        # Concatenating 10 scalar variables along axis 0 creates a (10,) shaped tensor.
        concatenated = tf.concat([tf.expand_dims(self.var, 0)] * 10, axis=0)
        return tf.reduce_mean(concatenated)

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a dummy scalar input tensor; the model doesn't actually use the input value.
    # Use shape=() scalar float tensor to match the fact that the computation doesn't depend on input shape.
    return tf.random.uniform((), dtype=tf.float32)

