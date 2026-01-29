# tf.random.uniform((batch_size, input_dim), dtype=tf.float32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, output_dim, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.flatten = tf.keras.layers.Flatten()
        self.hidden1 = tf.keras.layers.Dense(
            100, activation='elu', kernel_initializer='he_normal', name="hidden1"
        )
        self.resblock1 = self._make_resblock(2, 100)
        self.resblock2 = self._make_resblock(2, 100)
        self.output_layer = tf.keras.layers.Dense(
            output_dim, activation=tf.keras.activations.get(activation)
        )
        self._output_dim = output_dim
        self._activation = activation

    def _make_resblock(self, n_layers, n_neurons):
        # Define a residual block as a tf.keras.Layer with multiple Dense layers and skip connection
        class ResBlock(tf.keras.layers.Layer):
            def __init__(self, n_layers, n_neurons):
                super().__init__()
                self.hidden_layers = [
                    tf.keras.layers.Dense(
                        n_neurons,
                        activation='elu',
                        kernel_initializer='he_normal'
                    )
                    for _ in range(n_layers)
                ]
            def call(self, inputs):
                z = inputs
                for layer in self.hidden_layers:
                    z = layer(z)
                return inputs + z
        return ResBlock(n_layers, n_neurons)

    def call(self, inputs):
        z = self.flatten(inputs)
        z = self.hidden1(z)
        # Apply resblock1 four times
        for _ in range(4):
            z = self.resblock1(z)
        # Apply resblock2 once after that
        z = self.resblock2(z)
        return self.output_layer(z)

    def get_config(self):
        base_config = super().get_config()
        # Store output_dim and activation for potential serialization reuse
        return {**base_config,
                "output_dim": self._output_dim,
                "activation": self._activation}

def my_model_function():
    # Create a MyModel instance - use 10 output dim and softmax activation as default (typical classification)
    return MyModel(output_dim=10, activation='softmax')

def GetInput():
    # Assumptions:
    # Input to this model is arbitrary shape but must be compatible with Flatten layer.
    # From usage, likely image-like data with shape (batch_size, height, width, channels)
    # Let's assume batch size 4, and 28x28 grayscale images (e.g. MNIST-like) as a reasonable default.
    # dtype=float32.
    batch_size = 4
    height = 28
    width = 28
    channels = 1
    return tf.random.uniform((batch_size, height, width, channels), dtype=tf.float32)

