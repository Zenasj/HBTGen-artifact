# tf.random.uniform((B, input_dim)) ← Assuming 1D input tensor for CustomModel

import tensorflow as tf
from tensorflow import keras

class MyModel(tf.keras.Model):
    def __init__(self, hidden_units):
        super(MyModel, self).__init__()
        self.hidden_units = hidden_units
        self.dense_layers = [keras.layers.Dense(u) for u in hidden_units]

    def call(self, inputs):
        x = inputs
        for layer in self.dense_layers:
            x = layer(x)
        return x

    def get_config(self):
        # Return config dict for recreating the model
        return {"hidden_units": self.hidden_units}

    @classmethod
    def from_config(cls, config):
        # Instantiate the model from config dictionary
        return cls(**config)


def my_model_function():
    # Create an instance with some default hidden units configuration
    # [16, 16, 10] from the example in the issue for demonstration
    return MyModel([16, 16, 10])

def GetInput():
    # Generate input tensor matching the model input expected - 
    # The model expects a 2D tensor (batch_size, feature_dim)
    # Since no input_dim is explicitly given, infer from last Dense layer input:
    # The Dense layers work with input shape like (batch_size, input_dim).
    # We do not know input_dim, but can assume a feature dimension compatible with the first Dense layer.
    # Assuming feature_dim=20 (arbitrary reasonable choice)
    batch_size = 4
    input_dim = 20
    return tf.random.uniform((batch_size, input_dim), dtype=tf.float32)

