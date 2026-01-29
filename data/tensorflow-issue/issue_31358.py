# tf.random.uniform((B, 28, 28, 1), dtype=tf.float32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self._conv = tf.keras.layers.Conv2D(filters=64, kernel_size=5, activation=tf.nn.relu)
        self._batch_norm = tf.keras.layers.BatchNormalization()
        self._flatten = tf.keras.layers.Flatten()
        self._logits = tf.keras.layers.Dense(units=10)

    def call(self, inputs, training=False):
        # Reimplementing forward pass from original Model.forward with BatchNormalization training param.
        x = self._conv(inputs)
        x = self._batch_norm(x, training=training)  # training param avoids freezing issues
        x = self._flatten(x)
        logits = self._logits(x)
        return logits

def my_model_function():
    # Return an instance of the MyModel class
    return MyModel()

def GetInput():
    # Return a random input tensor with shape [batch, height, width, channels] matching MNIST format [N, 28, 28, 1]
    # Use float32 and values scaled similar to original (0-1 normalized)
    batch_size = 8  # A small batch size to ensure quick runs and compatibility
    input_tensor = tf.random.uniform(shape=(batch_size, 28, 28, 1), minval=0, maxval=1, dtype=tf.float32)
    return input_tensor

