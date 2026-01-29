# tf.random.uniform((1000, 28, 28), dtype=tf.float32) ← inferred from batch_size=1000 and input_shape=(28,28)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Simple fashion MNIST model as per original example
        # Flatten -> Dense(128, relu) -> Dense(10, softmax)
        self.flatten = tf.keras.layers.Flatten(input_shape=(28, 28))
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, inputs, training=False):
        x = self.flatten(inputs)
        x = self.dense1(x)
        return self.dense2(x)


def my_model_function():
    # Return an instance of MyModel.
    # Important usage note from the issue: when using
    # tf.distribute.Strategy (e.g., MirroredStrategy), model weights
    # and optimizer variables must be loaded/created inside strategy.scope().
    # Here we just return the model instance.
    return MyModel()


def GetInput():
    # Return a random input tensor matching model input shape: (batch_size, 28, 28)
    batch_size = 1000  # as per example
    # Use float32 uniformly distributed input between 0 and 1, compatible with normal preprocessing
    return tf.random.uniform((batch_size, 28, 28), dtype=tf.float32)

