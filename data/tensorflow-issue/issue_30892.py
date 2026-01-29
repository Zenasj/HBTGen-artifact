# tf.random.uniform((3, 32, 32, 3), dtype=tf.float32) ← inferred from example batch_size=3, image_shape=(32, 32, 3)
import tensorflow as tf


class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # A minimal Conv2D layer matching the example in issue:
        self.conv = tf.keras.layers.Conv2D(
            filters=16,
            kernel_size=3,
            strides=2,
            padding='SAME',
            activation='linear',
            input_shape=(32, 32, 3)  # Explicit input shape to avoid issues in saving/loading
        )
        # Initialize build by explicitly calling with sample input
        dummy_input = tf.zeros((1, 32, 32, 3))
        self.conv(dummy_input)

    def call(self, inputs):
        # Forward pass through the single Conv2D layer
        return self.conv(inputs)


def my_model_function():
    # Returns the MyModel instance, no pretrained weights are needed since this is a minimal example
    return MyModel()


def GetInput():
    # Returns a random input tensor with batch_size=3, height=32, width=32, channels=3 matching the model input shape
    return tf.random.uniform((3, 32, 32, 3), dtype=tf.float32)

