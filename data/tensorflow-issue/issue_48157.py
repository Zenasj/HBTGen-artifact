# tf.random.uniform((B, 32, 32, 128), dtype=tf.float32)
import tensorflow as tf
from collections import OrderedDict

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define inputs
        self.input_shape = (32, 32, 128)

        # Layers for boxes output: Conv2D with 4 filters, kernel 3x3
        self.conv_boxes = tf.keras.layers.Conv2D(4, (3,3), name="conv_boxes")
        
        # Layers for scores output: Conv2D with 1 filter, kernel 3x3
        self.conv_scores = tf.keras.layers.Conv2D(1, (3,3), name="conv_scores")

    def call(self, inputs):
        # Inputs is expected to be a dict with "inputs" key (matching the original model)
        if isinstance(inputs, dict):
            x = inputs.get("inputs", None)
            if x is None:
                raise ValueError("Input dictionary must contain 'inputs' key")
        else:
            # Allow raw tensor input
            x = inputs

        boxes = self.conv_boxes(x)
        # reshape so that the last dimension is 4, flatten batch dimensions accordingly
        boxes = tf.reshape(boxes, (tf.shape(boxes)[0], -1, 4))  # shape (batch, ?, 4)

        scores = self.conv_scores(x)
        # flatten scores similarly
        scores = tf.reshape(scores, (tf.shape(scores)[0], -1))  # shape (batch, ?)

        # Return dict matching keys as per issue discussion
        return OrderedDict([
            ("boxes", boxes),
            ("scores", scores)
        ])


def my_model_function():
    # Return a new instance of MyModel
    return MyModel()


def GetInput():
    # Returns a dict with "inputs" key matching what MyModel expects
    # Batch size 2 used as example
    batch_size = 2
    x = tf.random.uniform((batch_size, 32, 32, 128), dtype=tf.float32)
    return {"inputs": x}

