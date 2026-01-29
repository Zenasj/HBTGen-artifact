# tf.random.uniform((B, H, W, 3), dtype=tf.float32) ← Input batch with unknown spatial dimensions and 3 channels

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # We assume input shape is (None, None, 3) with unknown spatial dims
        # BatchNormalization with virtual_batch_size requires known batch dimension,
        # but original issue shows error when spatial dims are None.
        # We fix this by requiring a fixed input batch size or fallback behavior.
        # Here we assume batch size is known at runtime (not None).
        
        # We set momentum=0.9 (default), and virtual_batch_size=8 as per issue
        self.bn = tf.keras.layers.BatchNormalization(virtual_batch_size=8)
    
    def call(self, inputs, training=False):
        # inputs shape: (batch_size, height, width, 3)
        # We expect that batch_size and channel dimensions are known, height and width can be None.
        # Call BatchNormalization layer with virtual_batch_size.
        return self.bn(inputs, training=training)

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Batch size is required for virtual_batch_size to work. We set batch size=16 here (>=8).
    # Height and width are unknown during model build, but we provide concrete values here.
    batch_size = 16
    height = 32
    width = 32
    channels = 3
    # Generate random input tensor matching expected input shape.
    return tf.random.uniform((batch_size, height, width, channels), dtype=tf.float32)

