# tf.random.uniform((B, None, None, 3), dtype=tf.float32) ← Input is a batch of ragged images with channel=3

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # This model mimics the example architecture discussed:
        # Input: Ragged batch of images with shape (batch, height, width, 3)
        # Resize all images to (224, 224, 3)
        # Then Conv2D(24 filters, kernel 3x3) -> GlobalMaxPool2D -> Dense(1)
        
        self.resizing = tf.keras.layers.experimental.preprocessing.Resizing(
            224, 224, name="resize"
        )
        self.conv = tf.keras.layers.Conv2D(
            filters=24, kernel_size=3, activation='relu', name="kernel"
        )
        self.pool = tf.keras.layers.GlobalMaxPool2D(name="pool")
        self.dense = tf.keras.layers.Dense(1, name="dense_second")

    def call(self, inputs, training=False):
        # Since inputs may be a RaggedTensor with shape (batch, None, None, 3),
        # we first convert to dense with padding, as tf.image.resize requires dense tensors.
        # The downstream preprocessing and conv layers require dense tensors of fixed size.
        #
        # Assumption: Pad with zeros to max size in batch.
        
        if isinstance(inputs, tf.RaggedTensor):
            # Convert ragged images to dense, padding with zeros (0).
            # Shape becomes (batch, max_height, max_width, 3)
            inputs = inputs.to_tensor(default_value=0.0)
        
        x = self.resizing(inputs)
        x = self.conv(x)
        x = self.pool(x)
        x = self.dense(x)
        return x

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Returns a RaggedTensor batch of 2 images with variable height and width, 3 channels
    # Matching the example: one 512x512, another 1080x1920
    # Values uniformly random [0,1)
    a = tf.random.uniform(shape=(512, 512, 3), dtype=tf.float32)
    b = tf.random.uniform(shape=(1080, 1920, 3), dtype=tf.float32)
    ragged_images = tf.ragged.stack([a, b])  # shape: (2, None, None, 3)
    return ragged_images

