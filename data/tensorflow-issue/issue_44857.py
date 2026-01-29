# tf.random.uniform((1, 810, 1440, 3), dtype=tf.float32) ← Input batch size 1, image height 810, width 1440, 3 channels RGB

import tensorflow as tf
from tensorflow.keras.applications import DenseNet121

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Initialize DenseNet121 with specified input shape and imagenet weights, exclude top classification layers
        self.backbone = DenseNet121(input_shape=(810, 1440, 3),
                                    include_top=False,
                                    weights='imagenet')
        
        # The backbone model needs to be explicitly called at least once to build internal layers
        # so output shapes and weights are initialized.
        # We'll do this during the first call to MyModel.
        self._built = False

    def call(self, inputs, training=False):
        # On first call, call the backbone to build layers and define output shapes.
        if not self._built:
            _ = self.backbone(inputs, training=training)
            self._built = True
        
        # Forward the input through the DenseNet backbone and return its feature tensor output
        return self.backbone(inputs, training=training)

def my_model_function():
    # Create and return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random float32 input tensor matching the expected input shape of the DenseNet backbone
    # Typically models expect batch dimension; using batch size = 1 here.
    return tf.random.uniform((1, 810, 1440, 3), dtype=tf.float32)

