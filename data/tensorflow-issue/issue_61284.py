# tf.random.uniform((B, 3), dtype=tf.float32) ← Input shape inferred from Input(shape=(3)) in the issue

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Dense layer with 5 units and sigmoid activation as per example
        self.position_layer = tf.keras.layers.Dense(5, activation='sigmoid', name='position')

    def call(self, inputs):
        # Compute position output
        position = self.position_layer(inputs)

        # Compute a boolean tensor where position > 0.5
        position_bool = tf.cast(position > 0.5, tf.float32)

        # Sum along the last axis, keep dims, with name 'final' as in example
        out = tf.math.reduce_sum(position_bool, axis=-1, keepdims=True, name='final')

        # Return both position and reduced sum as outputs
        return position, out


def my_model_function():
    # Instantiate and return MyModel
    return MyModel()


def GetInput():
    # Return a random input tensor matching shape (batch_size, 3)
    # Batch size arbitrarily chosen as 4 for demonstration
    batch_size = 4
    return tf.random.uniform((batch_size, 3), dtype=tf.float32)

