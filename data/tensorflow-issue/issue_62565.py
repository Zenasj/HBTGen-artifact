# tf.random.uniform((1, H, W, 3), dtype=tf.float32) ← Input shape is dynamic height and width with 3 channels

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define the layers as described in the issue's model definition
        self.conv1 = tf.keras.layers.Conv2D(16, kernel_size=3, padding='same')
        self.conv2 = tf.keras.layers.Conv2D(32, kernel_size=3, padding='same', strides=2)
        self.conv3 = tf.keras.layers.Conv2D(64, kernel_size=3, use_bias=False)
        self.conv4 = tf.keras.layers.Conv2D(3, kernel_size=3, padding='same')
    
    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x

def my_model_function():
    # Create and return an instance of MyModel
    model = MyModel()
    # Optionally build the model by passing a dummy input with None for H and W
    dummy_input = tf.random.uniform((1, 16, 16, 3))
    model(dummy_input)  # build model weights
    return model

def GetInput():
    # Generate a valid random input tensor that matches the expected input of MyModel
    # Input shape: batch_size=1, height=variable, width=variable, channels=3
    # For demonstration, use height=512, width=512 matching the benchmark input shape
    input_shape = (1, 512, 512, 3)
    return tf.random.uniform(input_shape, dtype=tf.float32)

