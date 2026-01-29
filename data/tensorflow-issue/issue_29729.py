# tf.random.uniform((B, 2), dtype=tf.float32) ← based on input_shape=(2,) used in example

import tensorflow as tf
import numpy as np

class MyModel(tf.keras.Model):
    def __init__(self, state_dim=2, action_dim=2):
        super(MyModel, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Layer with input_shape specified, but in subclassed model this does not trigger variable creation
        self.fc1 = tf.keras.layers.Dense(100, input_shape=(self.state_dim,))
        self.fc2 = tf.keras.layers.Dense(1)
        
    def call(self, inputs):
        out = self.fc1(inputs)
        out = self.fc2(out)
        return out

def my_model_function():
    """
    Returns an instance of MyModel.
    This model is not fully initialized until first call or until build() is called with an input shape.
    """
    model = MyModel()
    # Optionally, force variable initialization by calling build with input shape
    # This mimics the advice in the issue that calling .build() triggers variable creation
    model.build(input_shape=(None, model.state_dim))
    return model

def GetInput():
    """
    Returns a random input tensor matching expected input shape of MyModel.
    Given fc1 layer expects input shape (batch_size, 2), generate accordingly.
    """
    batch_size = 10  # example batch size
    return tf.random.uniform((batch_size, 2), dtype=tf.float32)

