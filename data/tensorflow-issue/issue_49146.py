# tf.random.uniform((B, None, None, None, 1), dtype=tf.float32)  ← Input shape is variable 5D tensor (batch, depth, height, width, channels)

import tensorflow as tf
import numpy as np

class ExtractPatches(tf.keras.layers.Layer):
    def __init__(self, ksizes, strides, target_shape):
        super(ExtractPatches, self).__init__()
        self.ksizes = ksizes
        self.strides = strides
        self.target_shape = target_shape  # Expected output shape after reshaping patches

    def call(self, inputs):
        # Extract volume patches with variable input shape, returns reshaped patches and input shape
        patches = tf.extract_volume_patches(inputs,
                                           ksizes=self.ksizes,
                                           strides=self.strides,
                                           padding="VALID")
        # Reshape patches to the target shape (should match downstream conv3d input)
        reshaped = tf.reshape(patches, self.target_shape)
        return reshaped, tf.shape(inputs)

class CombinePatches(tf.keras.layers.Layer):
    def __init__(self, ksizes, strides):
        super(CombinePatches, self).__init__()
        self.ksizes = ksizes
        self.strides = strides

    def call(self, patches, inputs):
        # Initialize a zero volume of same shape & dtype as inputs
        target_volume = tf.zeros_like(inputs)
        # Extract patches from zero volume to get mapping shape
        target_patches = tf.extract_volume_patches(target_volume,
                                                   ksizes=self.ksizes,
                                                   strides=self.strides,
                                                   padding="VALID")
        # Compute gradient of extracted patches w.r.t. zero volume
        target_grad_mapping = tf.gradients(target_patches, target_volume)[0]

        # Compute gradient of patches w.r.t. zero volume divided by grad mapping
        # This approximates the inverse operation of extract_volume_patches,
        # averaging overlapping patches by dividing by number of overlaps
        combined = tf.gradients(patches, target_volume, patches)[0] / target_grad_mapping
        return combined

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Kernel and stride sizes for extract/combining patches
        self.ksizes = [1, 14, 14, 14, 1]
        self.strides = [1, 14, 14, 14, 1]
        # Placeholder shape for patch reshaping: (batch_size, patches_d, patches_h, patches_w, patch_size)
        # The patch_size corresponds to the flattened patch volume (14*14*14 * channels=1)
        # Since Conv3D expects 5D input, reshape should yield (batch, d, h, w, channels)
        # However, original example reshaped to (-1,14,14,14,1) but -1 used for batch
        # To adapt to variable batch size, use dynamic batch dimension

        # We can't statically infer patch counts without fixed input dims, so we use None for batch dimension
        # Assuming input channels = 1 in this setting
        self.patch_shape = (-1, 14, 14, 14, 1)

        self.extract_patches = ExtractPatches(self.ksizes, self.strides, self.patch_shape)
        self.combine_patches = CombinePatches(self.ksizes, self.strides)

        # Conv3D and Conv3DTranspose layers process patches and reconstruct
        self.encoder = tf.keras.layers.Conv3D(filters=28, kernel_size=(14,14,14), strides=(14,14,14))
        self.decoder = tf.keras.layers.Conv3DTranspose(filters=1, kernel_size=(14,14,14), strides=(14,14,14))

    def call(self, inputs):
        # Extract patches and get original input shape
        patches, input_shape = self.extract_patches(inputs)
        # Encode patches (3D conv)
        encoded = self.encoder(patches)
        # Decode (transpose conv)
        decoded = self.decoder(encoded)
        # Combine patches back into volume
        combined = self.combine_patches(decoded, inputs)
        return combined

def my_model_function():
    # Return an instance of the model
    return MyModel()

def GetInput():
    # Generate a sample input tensor with batch size 1, variable depth/height/width (multiple of 14 + 0 for valid extract)
    # Using batch 1 and shape (28, 28, 28, 1) which aligns with patch size and stride of 14
    batch_size = 1
    depth = 28
    height = 28
    width = 28
    channels = 1
    # Random uniform tensor mimicking the 3D image input (float32)
    input_tensor = tf.random.uniform((batch_size, depth, height, width, channels), dtype=tf.float32)
    return input_tensor

