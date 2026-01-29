# tf.random.uniform((B, 8), dtype=tf.float32) ← Input shape inferred from X_train shape (1000, 8)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, output_dim=1):
        super().__init__()
        self.hidden = tf.keras.layers.Dense(30, activation="elu")
        self.out = tf.keras.layers.Dense(output_dim)
        self.reconstruct = None  # Will be created in build()

    def build(self, input_shape):
        # input_shape is (batch_size, features)
        n_inputs = input_shape[-1]
        # Create the reconstruct layer dynamically based on input feature size
        self.reconstruct = tf.keras.layers.Dense(n_inputs)
        super().build(input_shape)

    def call(self, inputs):
        Z = self.hidden(inputs)
        # Run the dynamically created reconstruction layer
        reconstruction = self.reconstruct(Z)
        # Compute reconstruction loss as mean squared error between reconstruction and input
        reconstruction_loss = tf.reduce_mean(tf.square(reconstruction - inputs))
        # Add this as a weighted loss to the model
        self.add_loss(0.1 * reconstruction_loss)
        return self.out(Z)


def my_model_function():
    # Return an instance of MyModel with output dimension 1 (regression output)
    return MyModel(output_dim=1)


def GetInput():
    # Generate a batch of input data compatible with model input shape
    # Using batch size = 4 arbitrarily, feature dim = 8 as in example
    return tf.random.uniform((4, 8), dtype=tf.float32)

