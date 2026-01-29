# tf.random.uniform((B, 28, 28, 1), dtype=tf.float32) 
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Following the example sequential model from the issue description
        self.conv = tf.keras.layers.Conv2D(32, 3, activation="relu", input_shape=(28, 28, 1))
        self.pool = tf.keras.layers.MaxPooling2D()
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(64, activation="relu")
        self.dense2 = tf.keras.layers.Dense(10, activation="softmax")

    def call(self, inputs, training=False):
        x = self.conv(inputs)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return x

def my_model_function():
    # Return an instance of MyModel; no weights loading needed for demonstration
    return MyModel()

def GetInput():
    # Return a batch of images matching input shape (batch_size, 28, 28, 1)
    # Use batch size 256 as in the issue example for model.fit
    batch_size = 256
    # Generate random float input in [0,1) to simulate normalized MNIST images
    return tf.random.uniform((batch_size, 28, 28, 1), dtype=tf.float32)

