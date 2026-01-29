# tf.random.uniform((10, 10), dtype=tf.float32) ← input shape inferred from dataset_fn's x

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Single dense layer model as per the original code's Sequential model with one Dense(10)
        self.dense = tf.keras.layers.Dense(10)
        
    def call(self, inputs, training=False):
        # Forward pass through the dense layer
        return self.dense(inputs)

def my_model_function():
    """
    Returns an instance of MyModel.
    This matches the model created inside the ParameterServerStrategy scope.
    """
    return MyModel()

def GetInput():
    """
    Returns a random input tensor matching the expected input shape (10,10).
    The dtype is float32 to be compatible with the Dense layer.
    """
    return tf.random.uniform((10, 10), dtype=tf.float32)

