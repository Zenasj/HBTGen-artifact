# tf.random.uniform((None, 28, 28), dtype=tf.uint8) ← Input shape inferred from MNIST dataset used in example

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Model architecture as given in minimal reproducible example:
        # Reshape input to (?, -1, 784), normalize dividing by 255, 2 Dense layers
        self.reshape = tf.keras.layers.Reshape((-1, 784))  # Batch size dynamic, flatten 28x28 to 784
        self.normalize = tf.keras.layers.Lambda(lambda x: tf.divide(tf.cast(x, tf.float32), 255.))
        # Important: Use proper initializer instance, not the class or function reference without calling it
        # Following best practice as per the issue: use tf.initializers.Zeros() rather than tf.initializers.zeros or the class tf.initializers.Zeros
        self.dense1 = tf.keras.layers.Dense(
            256, activation='relu', bias_initializer=tf.initializers.Zeros()
        )
        self.dense2 = tf.keras.layers.Dense(
            10, activation='softmax'
        )
        
    def call(self, inputs):
        x = self.reshape(inputs)
        x = self.normalize(x)
        x = self.dense1(x)
        return self.dense2(x)


def my_model_function():
    # Instantiate and return the model
    return MyModel()


def GetInput():
    # Return a random tensor simulating MNIST images:
    # MNIST images shape: (batch_size, 28, 28), original dtype = uint8 as pixel intensities 0-255
    # Use batch size 4 (arbitrary) for example
    batch_size = 4
    return tf.random.uniform(shape=(batch_size,28,28), maxval=256, dtype=tf.uint8)


# Note:
# This code reflects the core example from the issue illustrating the problem and fix.
# It uses the proper initializer instance tf.initializers.Zeros(), not the class or function reference without calling it,
# which was the root cause of serialization failing.
# The input shape is assumed to be (batch,28,28) uint8 images as in MNIST.
# The model can be compiled, trained, and saved without serialization issues.

