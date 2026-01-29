# tf.random.uniform((B, 512), dtype=tf.float32) ← B is batch size, input vectors are 512-dim embeddings

import tensorflow as tf
import numpy as np

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define Dense layer branches for two 512-dim inputs
        self.branch1_dense1 = tf.keras.layers.Dense(128, activation="relu")
        self.branch2_dense1 = tf.keras.layers.Dense(128, activation="relu")
        
        # Combined layers: note original code had an error where 'z' layer was incorrectly applied on combined twice,
        # We'll replicate that faithfully and comment accordingly:
        self.combined_dense1 = tf.keras.layers.Dense(16, activation="relu")
        self.combined_dense2 = tf.keras.layers.Dense(4, activation="relu")
        self.output_layer = tf.keras.layers.Dense(2, activation="linear")

        self.concat = tf.keras.layers.Concatenate()

    def call(self, inputs):
        # inputs is a list or tuple of two tensors: (x1, x2)
        x1, x2 = inputs
        
        x = self.branch1_dense1(x1)  # shape (B,128)
        y = self.branch2_dense1(x2)  # shape (B,128)

        combined = self.concat([x, y])  # shape (B,256)

        # According to supplied snippet, the second dense is applied on combined instead of the output of first dense:
        # That is a mistake but to replicate exactly:
        z = self.combined_dense1(combined)
        z = self.combined_dense2(combined)  # overwrites previous z
        out = self.output_layer(z)
        return out


def my_model_function():
    # Create and compile the model similarly to original sample code:
    model = MyModel()
    # Compile with Adam optimizer and mean_absolute_percentage_error loss as used in the snippet:
    opt = tf.keras.optimizers.Adam(learning_rate=1e-3, decay=1e-3 / 200)
    model.compile(optimizer=opt, loss="mean_absolute_percentage_error")
    return model

def GetInput():
    # Return a batch of random input tensors matching expected input shape,
    # assume batch size = 16 to match batch_size in generator in original code.
    batch_size = 16
    # Generate random float tensor inputs shaped (batch_size, 512)
    x1 = tf.random.uniform((batch_size, 512), dtype=tf.float32)
    x2 = tf.random.uniform((batch_size, 512), dtype=tf.float32)
    return [x1, x2]

