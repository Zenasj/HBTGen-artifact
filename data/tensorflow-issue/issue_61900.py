# tf.random.uniform((B, 1), dtype=tf.float32) ← the original input x shape feeding into a model with input_shape=(1,)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Layer sizes based on original model:
        # Input shape: single feature (scalar per example)
        # Dense units: 25 → 15 → 1 (final output)
        self.layer1 = tf.keras.layers.Dense(units=25, activation='relu', name='layer1')
        self.layer2 = tf.keras.layers.Dense(units=15, activation='relu', name='layer2')
        self.layer3 = tf.keras.layers.Dense(units=1, activation='linear', name='layer3')

    def call(self, inputs, training=False):
        # Forward pass through the network
        x = self.layer1(inputs)
        x = self.layer2(x)
        output = self.layer3(x)
        return output


def my_model_function():
    # Returns a fresh instance of MyModel
    return MyModel()


def GetInput():
    # Construct random inputs consistent with the original shape:
    # The original x was a list of shape (4,1), so batch size variable is fine.
    # We use float32 as dtype since TF default is float32 for layers.
    # Note: batch size 4, feature dimension 1 is assumed here.
    return tf.random.uniform(shape=(4, 1), dtype=tf.float32)

