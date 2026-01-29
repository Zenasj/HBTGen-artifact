# tf.random.uniform((B, N), dtype=tf.float32) ← Assuming input is a batch of feature vectors with shape (batch_size, feature_dim)

import tensorflow as tf

class MyModel(tf.keras.Model):
    """
    A reusable MLP model implementing a sequence of Dense layers each followed by
    BatchNormalization, ReLU activation, and Dropout, except the final Dense layer
    which is optionally followed by Dropout only.

    This corresponds to the example MLP discussed in the issue, encapsulated as a tf.keras.Model.
    """

    def __init__(self, hidden_channels=[128, 64, 32, 10], 
                 norm_layer=tf.keras.layers.BatchNormalization, 
                 activation_layer=tf.keras.layers.ReLU, 
                 dropout=0.2,
                 **kwargs):
        super().__init__(**kwargs)
        self.layers_list = []

        # Build MLP layers except the last output layer
        for units in hidden_channels[:-1]:
            self.layers_list.append(tf.keras.layers.Dense(units))
            self.layers_list.append(norm_layer())
            self.layers_list.append(activation_layer())
            self.layers_list.append(tf.keras.layers.Dropout(dropout))
        # Add final Dense layer
        self.layers_list.append(tf.keras.layers.Dense(hidden_channels[-1]))
        # Add Dropout after final layer too (as per original example)
        self.layers_list.append(tf.keras.layers.Dropout(dropout))

    def call(self, inputs, training=False):
        x = inputs
        for layer in self.layers_list:
            # For layers that use training argument (BatchNorm, Dropout), pass training flag
            if isinstance(layer, (tf.keras.layers.BatchNormalization, tf.keras.layers.Dropout)):
                x = layer(x, training=training)
            else:
                x = layer(x)
        return x


def my_model_function():
    """
    Returns an instance of MyModel with default parameters equivalent to the example:
    hidden_channels=[128, 64, 32, 10],
    norm_layer=tf.keras.layers.BatchNormalization,
    activation_layer=tf.keras.layers.ReLU,
    dropout=0.2
    """
    return MyModel()


def GetInput():
    """
    Generates a random input tensor with shape (batch_size, feature_dim) compatible with MyModel.

    Since the input dimension is not explicitly stated in the issue, we assume a typical input
    size, for example 100 features, and batch size 4 for testing.
    """
    batch_size = 4
    feature_dim = 100  # Assumed input dimension
    # Uniform random floats in [0,1), dtype float32
    return tf.random.uniform((batch_size, feature_dim), dtype=tf.float32)

