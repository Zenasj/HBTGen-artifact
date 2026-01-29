# tf.random.uniform((B, 5, 5, 3), dtype=tf.float32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Following the example, the model consists of:
        # Input shape: (None, 5, 5, 3)
        # Conv2D with 32 filters, kernel 3x3, stride 2, padding "same"
        # followed by BatchNormalization
        self.conv = tf.keras.layers.Conv2D(
            filters=32, kernel_size=3, strides=2, padding="same")
        self.bn = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training=False):
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        return x

def my_model_function():
    # Returns a MyModel instance with default initialization
    # This matches the original Keras model structure in the issue.
    return MyModel()

def GetInput():
    # Generate input tensors shaped (batch, height=5, width=5, channels=3)
    # The exact batch size is not specified in the original code.
    # For typing and practical testing, we use batch size = 1.
    # dtype float32 as per typical Keras inputs.
    return tf.random.uniform(shape=(1, 5, 5, 3), dtype=tf.float32)

