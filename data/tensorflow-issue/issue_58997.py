# tf.random.uniform((B, 28, 28, 1), dtype=tf.float32) ← Input shape corresponds to Fashion MNIST grayscale images (28x28x1)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Following the sequential model architecture described in the issue:
        # Conv2D with 32 filters, kernel 3x3, input shape (28,28,1)
        self.conv = tf.keras.layers.Conv2D(32, (3, 3), activation=None, input_shape=(28, 28, 1))
        # Flatten layer
        self.flatten = tf.keras.layers.Flatten()
        # Dense output layer with 10 units and softmax activation
        self.dense = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, x, training=False):
        x = self.conv(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x

def my_model_function():
    # Return an instance of MyModel.
    # There is no mention of loading saved weights in the issue for this extraction,
    # so we instantiate a fresh model.
    # Note: For replicating training or loading weights, additional code can be added.
    return MyModel()

def GetInput():
    # Return a random input tensor matching the shape expected by the model.
    # Since the inputs in the original context are normalized image batches from Fashion MNIST,
    # shape: (batch_size, 28, 28, 1), dtype float32, values in [0,1].
    # We pick batch size 8 as a reasonable example.
    batch_size = 8
    return tf.random.uniform((batch_size, 28, 28, 1), minval=0.0, maxval=1.0, dtype=tf.float32)

