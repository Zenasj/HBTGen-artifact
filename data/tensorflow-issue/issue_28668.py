# tf.random.uniform((batch_size, 32, 32, 3), dtype=tf.float32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # We define a simple Conv2D-based Sequential-style model
        # Matching the example: input shape (32,32,3), conv2d filters=16, kernel_size=3, strides=2, padding='SAME'
        self.conv = tf.keras.layers.Conv2D(
            filters=16,
            kernel_size=3,
            strides=2,
            padding='SAME',
            activation='linear',
            input_shape=(32, 32, 3)
        )

    def call(self, inputs):
        # Forward pass through Conv2D layer
        return self.conv(inputs)


def my_model_function():
    # Instantiate and return the model instance
    # Note: Because the model requires input shape specification for proper save/load, input_shape is set in layer init
    model = MyModel()
    # Create a dummy forward pass to build the model weights
    dummy_input = tf.zeros((1, 32, 32, 3), dtype=tf.float32)
    _ = model(dummy_input)
    return model


def GetInput():
    # Return a random tensor mimicking an input batch to the model
    batch_size = 3  # example batch size as per the issue examples
    input_shape = (32, 32, 3)
    # Create a batch of inputs with uniform random values
    return tf.random.uniform((batch_size, *input_shape), dtype=tf.float32)

