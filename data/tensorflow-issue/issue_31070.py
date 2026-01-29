# tf.random.uniform((None, 128, 128, 1), dtype=tf.float32) ← Assumed input shape from the model definition (128x128 grayscale images)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Define the same layers as in the original simple model used in the issue,
        # following the sequential architecture described:
        # Conv2D(32, 3, relu, padding='same', L2 reg 0.04)
        # Conv2D(1, 3, relu, padding='same', L2 reg 0.04)
        # Dense(1, softmax) - Note original code likely erred here: Dense after Conv2D without flatten or time-distributed.
        # To keep behavior, we use Conv2D + Conv2D + a 1x1 Conv2D to simulate Dense(1) per pixel with softmax activation.
        
        l2_reg = tf.keras.regularizers.l2(0.04)
        self.conv1 = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same', kernel_regularizer=l2_reg)
        self.conv2 = tf.keras.layers.Conv2D(1, 3, activation='relu', padding='same', kernel_regularizer=l2_reg)
        # Instead of Dense applied spatially, use Conv2D(1x1) with softmax over channels (not typical)
        # But softmax over single channel useless; likely a sigmoid was intended.
        # To reproduce faithfully, use Conv2D with softmax activation channel-wise.
        self.final_conv = tf.keras.layers.Conv2D(1, 1, activation='softmax')  # Approximate original intention

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.final_conv(x)
        return x


def my_model_function():
    # Return an instance of MyModel
    return MyModel()


def GetInput():
    # Returns a random tensor suitable as input to MyModel
    # Shape is (batch_size, 128, 128, 1), dtype float32
    # Use batch_size = 4 as a reasonable default for testing
    batch_size = 4
    return tf.random.uniform(shape=(batch_size, 128, 128, 1), dtype=tf.float32)

