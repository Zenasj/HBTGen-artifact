# tf.random.uniform((1, 224, 224, 3), dtype=tf.float32) ← inferred input shape from MobileNetV2 usage

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Load the official MobileNetV2 model with include_top=False to match the issue's scenario
        # Using pretrained imagenet weights, matching the example.
        self.mobilenet = tf.keras.applications.MobileNetV2(
            weights="imagenet", input_shape=(224, 224, 3), include_top=False)

    def call(self, inputs, training=False):
        # Forward pass through the MobileNetV2 feature extractor
        return self.mobilenet(inputs, training=training)

def my_model_function():
    # Returns an instance of MyModel initialized with pretrained MobileNetV2 weights
    return MyModel()

def GetInput():
    # Returns a random input tensor of shape [1, 224, 224, 3] in float32,
    # matching the MobileNetV2 input expected shape from the issue example.
    return tf.random.uniform((1, 224, 224, 3), dtype=tf.float32)

