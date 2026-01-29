# tf.random.uniform((1, 1, 126), dtype=tf.float32) ← inferred input shape based on the issue example

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # The model contains a single Reshape layer which reshapes the input from (1, 1, 126)
        # to (1, -1, 21). The -1 here means the dimension is inferred, expected 6 given input of 126.
        # This is the core model from the issue.
        self.reshape = tf.keras.layers.Reshape(target_shape=(-1, 21))

    def call(self, inputs, training=False):
        # Forward pass applies the reshape layer
        return self.reshape(inputs)

def my_model_function():
    # Instantiate and return the MyModel instance
    model = MyModel()
    # Model weights are untrained (random); the example issue runs one epoch of fit, 
    # but no saved weights provided. So we return fresh instance.
    return model

def GetInput():
    # Return a random (1, 1, 126) float32 tensor as input matching the model input signature
    # This matches the input shape in the original code:
    # inputs = np.random.rand(126).reshape((1, 1, 126))
    return tf.random.uniform(shape=(1, 1, 126), dtype=tf.float32)

