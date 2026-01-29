# tf.random.uniform((B, H, W, 3), dtype=tf.float32) ← Assuming input images with 3 color channels, height and width are dynamic

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Based on the issue, tf.image.extract_patches does NOT support RaggedTensors directly.
        # So we must flatten the ragged dimensions or reject ragged inputs.
        # Here we assume input is a dense tensor of shape [batch, height, width, 3].
        # The patch extraction uses sizes=[1,4,4,1], strides=[1,4,4,1], rates=[1,1,1,1], padding="SAME"

        self.sizes = [1, 4, 4, 1]
        self.strides = [1, 4, 4, 1]
        self.rates = [1, 1, 1, 1]
        self.padding = "SAME"

    def call(self, inputs):
        # Inputs should be a dense 4D tensor: [batch, height, width, channels=3]
        # If given a ragged tensor, extract_patches throws an error.
        # We add a runtime check to reject ragged input.

        if isinstance(inputs, tf.RaggedTensor):
            raise TypeError(
                "tf.image.extract_patches does not support RaggedTensor inputs. "
                "Convert input to dense tensor before calling the model."
            )
        
        # Extract patches using tf.image.extract_patches
        patches = tf.image.extract_patches(
            images=inputs,
            sizes=self.sizes,
            strides=self.strides,
            rates=self.rates,
            padding=self.padding,
        )
        return patches


def my_model_function():
    # Create an instance of MyModel 
    return MyModel()


def GetInput():
    # Provide a dense input tensor compatible with MyModel
    # Since patch size is 4x4 with stride 4, image size should be at least 8x8 for multiple patches.
    # Batch size=2 for example; channels=3 RGB
    batch_size = 2
    height = 8
    width = 8
    channels = 3
    # Random float32 tensor simulating batch of images
    input_tensor = tf.random.uniform(shape=(batch_size, height, width, channels), dtype=tf.float32)
    return input_tensor

