# tf.random.uniform((B, 224, 224, 3), dtype=tf.float32) ← inferred input shape from ResNet50 input

import tensorflow as tf

class MyModel(tf.keras.Model):
    """
    A wrapper model encapsulating ResNet50.

    This class demonstrates the kind of input/output shape and usage close to the example in the issue.
    """

    def __init__(self):
        super().__init__()
        # Instantiate ResNet50 from keras applications with default weights None for simplicity and reproducibility
        # Input shape is (224, 224, 3), output shape is (1000,)
        self.resnet50 = tf.keras.applications.ResNet50(weights=None)

    def call(self, inputs, training=False):
        # Forward pass through ResNet50
        return self.resnet50(inputs, training=training)

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel (batch size 16 assumed)
    # The original code used batch size of 16 in the dataset example
    batch_size = 16
    input_shape = (batch_size, 224, 224, 3)
    # Use uniform random floats as dummy input similar to zeros tensor in original code
    return tf.random.uniform(input_shape, dtype=tf.float32)

