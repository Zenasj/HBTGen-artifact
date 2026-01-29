# tf.random.uniform((1,), dtype=tf.float32) ← Input shape inferred from the single scalar variable initialization

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Model has a single trainable scalar variable initialized to 1.0
        # Use a resource variable to match the distributed strategy behavior described
        self.var = tf.Variable(initial_value=1.0, dtype=tf.float32, trainable=True, name="var", use_resource=True)

    def call(self, inputs):
        # Forward pass just returns the variable (simulating the loss as in the original code)
        # Inputs are not used as the original TF estimator model_fn ignored features except for variable update
        return self.var

def my_model_function():
    # Return an instance of MyModel with trainable variable initialized to 1.0
    return MyModel()

def GetInput():
    # Return a dummy tensor input to satisfy the call interface of MyModel
    # Original estimator takes some features and labels, but model_fn ignores them
    # In Keras Model, we expect some tensor input; since var doesn't use input, shape can be minimal
    return tf.random.uniform((1,), dtype=tf.float32)

