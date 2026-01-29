# tf.random.uniform((B, 512, 512, 1), dtype=tf.float32) ← Input shape inferred from Keras Input(shape=(512, 512, 1))

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define layers consistent with the issue's example model
        # Conv2D with 64 filters, 3x3 kernel, no activation, no bias, padding='same'
        self.conv2d = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=(3, 3),
            activation=None,
            padding='same',
            use_bias=False,
            name="conv2d"
        )
        # Conv2DTranspose with 32 filters, 3x3 kernel, strides=2, padding='same', ReLU activation
        self.conv2d_transpose = tf.keras.layers.Conv2DTranspose(
            filters=32,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding='same',
            activation=tf.nn.relu,
            name="conv2d_transpose"
        )

    def call(self, inputs, training=False):
        # Forward pass matching the example: Input → Conv2D → Conv2DTranspose
        x = self.conv2d(inputs)
        output = self.conv2d_transpose(x)
        return output

def my_model_function():
    # Return an instance of MyModel
    model = MyModel()
    # Weights are not initialized from checkpoint here, user can load from "results/checkpoint" if needed.
    return model

def GetInput():
    # Return random input tensor matching the Keras Input shape (batch size 1, 512, 512, 1)
    # Batch size is arbitrary, set to 1 for convenience
    input_tensor = tf.random.uniform(shape=(1, 512, 512, 1), dtype=tf.float32)
    return input_tensor

