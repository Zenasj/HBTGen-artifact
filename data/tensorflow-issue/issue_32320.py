# tf.random.uniform((B, 28, 28), dtype=tf.float32) ← Input shape inferred as batches of 28x28 grayscale images

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # The model is a classical MNIST/Fashion-MNIST style classifier:
        # Flatten input 28x28 to 784 vector, dense 128 relu, dense 10 softmax output
        self.flatten = tf.keras.layers.Flatten(input_shape=(28, 28))
        self.dense1 = tf.keras.layers.Dense(128, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(10, activation=tf.nn.softmax)

    def call(self, inputs, training=False):
        x = self.flatten(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        return x

def my_model_function():
    # Return a fresh instance of the model with uninitialized weights
    return MyModel()

def GetInput():
    # Return a batch of random input images with shape (batch_size=32, 28, 28)
    # Use float32 in [0,1) range to mimic normalized grayscale images (like Fashion-MNIST)
    batch_size = 32  # typical batch size chosen as a reasonable default
    return tf.random.uniform((batch_size, 28, 28), minval=0, maxval=1, dtype=tf.float32)

