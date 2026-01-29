# tf.random.uniform((B, 128, 128, 1), dtype=tf.float32) ← Input shape inferred from issue's example

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Following the example architecture in the issue:
        # Input shape (128, 128, 1)
        # Conv2D layers: 32 filters -> 64 filters -> 1 filter with sigmoid activation
        self.conv1 = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')
        self.conv2 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')
        self.conv3 = tf.keras.layers.Conv2D(1, 3, activation='sigmoid', padding='same')

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.conv2(x)
        output = self.conv3(x)
        return output

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Generate a random float32 tensor with shape (batch_size, 128, 128, 1)
    # Here batch_size is arbitrary, assume 1 for demonstration
    return tf.random.uniform((1, 128, 128, 1), dtype=tf.float32)

