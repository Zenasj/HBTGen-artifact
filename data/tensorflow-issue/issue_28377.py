# tf.random.uniform((B, input_dim), dtype=tf.float32) ← input shape is (batch_size, input_dim), unknown input_dim but can be inferred as feature dimension from input

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, input_dim, layers=None):
        """
        Build dropout FCNN with two outputs for aleatoric and epistemic uncertainty estimation.
        Args:
            input_dim (int): number of features in the input (columns).
            layers (tuple): optional layer specification overriding the default.
        """
        super().__init__()
        default_layers = (
            ("Dense", {"units": 100, "activation": "tanh"}),
            ("Dropout", {"rate": 0.5}),
            ("Dense", {"units": 50, "activation": "relu"}),
            ("Dropout", {"rate": 0.3}),
            ("Dense", {"units": 25, "activation": "relu"}),
            ("Dropout", {"rate": 0.3}),
            ("Dense", {"units": 10, "activation": "relu"}),
            ("Dropout", {"rate": 0.3}),
        )
        self.input_layer = tf.keras.layers.InputLayer(input_shape=(input_dim,), name="input")
        self.hidden_layers = []
        for layer_name, kwargs in layers or default_layers:
            layer = getattr(tf.keras.layers, layer_name)(**kwargs)
            self.hidden_layers.append(layer)
        # Final output layer with 2 outputs: predictive mean and log variance (aleatoric uncertainty)
        self.output_layer = tf.keras.layers.Dense(units=2, activation="linear", name="output")
    
    def call(self, inputs, training=False):
        x = self.input_layer(inputs)
        for layer in self.hidden_layers:
            x = layer(x, training=training)
        output = self.output_layer(x)
        # output shape: (batch_size, 2) where
        # output[:,0] = predictive mean
        # output[:,1] = predicted log variance (uncertainty)
        return output

def my_model_function():
    """
    Returns an instance of MyModel.
    NOTE: The input_dim here needs to be set according to actual feature dimension.
    For generically runnable code, we'll assume input_dim=10 as a placeholder.
    Modify accordingly when known.
    """
    input_dim = 10  # Assumed default input dimension; change as needed
    return MyModel(input_dim=input_dim)

def GetInput():
    """
    Returns a random input tensor matching the expected input of MyModel.
    Assumes input_dim=10 as per my_model_function.
    
    Generates a batch of 32 samples.
    """
    batch_size = 32
    input_dim = 10  # Must match MyModel's input_dim
    # random floats in [0,1)
    return tf.random.uniform((batch_size, input_dim), dtype=tf.float32)

