# tf.random.uniform((16, 24, 24, 3), dtype=tf.float32) ← inferred input shape from provided code
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Reconstructing the sequential model from the issue description
        self.conv = tf.keras.layers.Conv2D(8, kernel_size=(5, 5), padding='same', input_shape=(24, 24, 3))
        self.pool = tf.keras.layers.MaxPooling2D(pool_size=(24, 24), padding='valid')
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(3)

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.pool(x)
        x = self.flatten(x)
        return self.dense(x)

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random tensor input matching the expected input shape (batch size 16)
    # The original code uses batch=16, 24x24 RGB images of float32
    return tf.random.uniform((16, 24, 24, 3), dtype=tf.float32)

