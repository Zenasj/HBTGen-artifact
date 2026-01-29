# tf.random.uniform((1, 3, 3, 3, 4), dtype=tf.float64) ← Inferred input shape and dtype from issue examples

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Using AveragePooling3D with pool_size=(3,1,3) matching the issue example input
        self.avg_pool = tf.keras.layers.AveragePooling3D(pool_size=(3, 1, 3))
        # Using MaxPool3D with pool_size=(1, 2, 2) from the second example in the issue
        self.max_pool = tf.keras.layers.MaxPool3D(pool_size=(1, 2, 2))

    def call(self, inputs):
        # Inputs assumed to be float64 tensor with shape (batch, depth, height, width, channels)
        # Apply both pooling layers to the input
        avg_out = self.avg_pool(inputs)
        max_out = self.max_pool(inputs)

        # Since there's no original "comparison" in the issue, we fuse outputs by concatenation
        # along channels axis for demonstration. This keeps info from both layers.
        # Note: Shapes may differ due to different pool sizes, so pad if needed or return as tuple.
        return avg_out, max_out

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random float64 tensor with shape (1, 3, 3, 3, 4)
    # Matches the input_shape=(3,3,3,4) of the AveragePooling3D layer in the issue,
    # and batch size = 1 as in code example.
    return tf.random.uniform(
        shape=(1, 3, 3, 3, 4), dtype=tf.float64, minval=0, maxval=10
    )

