# tf.random.uniform((32, 28, 28), dtype=tf.float32) ← Based on example input shape in the issue

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Reconstruct the small_model (an intermediate sub-model)
        inputs = tf.keras.layers.Input(shape=(28, 28))
        x = tf.keras.layers.Flatten()(inputs)
        x = tf.keras.layers.Dense(128)(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Dense(10)(x)
        self.small_model = tf.keras.Model(inputs=inputs, outputs=x)

        # Big model takes same input as small_model and passes through small_model then extra layers
        big_input = self.small_model.input
        big_x = self.small_model.output
        big_x = tf.keras.layers.Dense(20)(big_x)
        big_x = tf.keras.layers.Dense(30)(big_x)
        big_x = tf.keras.layers.Dense(10)(big_x)
        self.big_model = tf.keras.Model(inputs=big_input, outputs=big_x)

    def call(self, inputs):
        # Forward through big_model and also capture the output of the small_model inside
        small_out = self.small_model(inputs)
        big_out = self.big_model(inputs)

        # The issue discussed is about gradients w.r.t outputs of a sub-model (small_model)
        # Return both outputs so that gradient can be checked externally
        return big_out, small_out


def my_model_function():
    return MyModel()

def GetInput():
    # From the issue example: input shape is (batch_size=32, 28, 28), dtype float32
    return tf.random.uniform((32, 28, 28), dtype=tf.float32)

