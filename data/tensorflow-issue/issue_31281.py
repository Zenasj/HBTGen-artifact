# tf.random.uniform((B, 15), dtype=tf.float32) ← input shape inferred from Dense layer input_dim=15

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, layer_density=64, layers=3, learning_rate=0.001):
        super().__init__()
        # This model replicates the described Keras Sequential model with Dense layers
        # and 'selu' activations as per the user's sample code snippet.
        # The output layer has 9 units with softmax activation.

        self.hidden_layers = []
        for _ in range(layers):
            self.hidden_layers.append(tf.keras.layers.Dense(layer_density, activation='selu'))
        self.output_layer = tf.keras.layers.Dense(9, activation='softmax', name='Output')
        # We will store optimizer internally for completeness, though not usually in model def.
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    def call(self, inputs, training=False):
        x = inputs
        for layer in self.hidden_layers:
            x = layer(x)
        return self.output_layer(x)

def my_model_function():
    # Instantiate MyModel with default parameters similar to example:
    # layer_density=64, layers=3, learning_rate=0.001 (arbitrary reasonable defaults)
    return MyModel(layer_density=64, layers=3, learning_rate=0.001)

def GetInput():
    # Return random input tensor matching the expected input shape:
    # batch size B unknown, so choose a batch of 32 as a standard default,
    # and 15 features as per the input_dim of first Dense layer.
    return tf.random.uniform((32, 15), dtype=tf.float32)

