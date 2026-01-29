# tf.random.uniform((B=1, H=32, W=32, C=3), dtype=tf.float32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Input shape inferred from example: (32,32,3)
        # We include a BatchNormalization layer as per the example reproducing the issue.
        self.bn = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training=False):
        # Forward pass through BatchNormalization layer
        return self.bn(inputs, training=training)

def my_model_function():
    # Return an instance of MyModel with default initialization.
    return MyModel()

def GetInput():
    # Return a random input tensor matching expected shape (batch, height, width, channels)
    # Using batch size 1 as example.
    return tf.random.uniform((1, 32, 32, 3), dtype=tf.float32)

