# tf.random.uniform((None, 28, 28, 1), dtype=tf.float32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define layers similar to the provided MNIST example model
        self.conv = tf.keras.layers.Conv2D(32, 3, activation="relu", input_shape=(28, 28, 1))
        self.pool = tf.keras.layers.MaxPooling2D()
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(64, activation="relu")
        self.dense2 = tf.keras.layers.Dense(10)

    def call(self, inputs, training=False):
        x = self.conv(inputs)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.dense1(x)
        return self.dense2(x)

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random input tensor simulating a batch of grayscale 28x28 images
    # The batch size is chosen as 64 like in the example training batch size
    batch_size = 64
    return tf.random.uniform((batch_size, 28, 28, 1), dtype=tf.float32)

