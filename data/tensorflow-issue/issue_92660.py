# tf.random.uniform((B, 128, 128, 128, 1), dtype=tf.float32)

import tensorflow as tf
from tensorflow.keras import layers, models
import tensorflow_addons as tfa

class SpatialTransformer(layers.Layer):
    """
    3D Spatial Transformer that applies a dense warp to a 5D volume tensor [B,D,H,W,C]
    using a flow field of shape [B,D,H,W,3].
    Internally performs batched 2D dense image warp on flattened depth slices for efficiency.
    """
    def call(self, inputs):
        vol, flow = inputs  # vol: [B, D, H, W, C], flow: [B, D, H, W, 3]

        # Enforce static known shapes (except batch) to satisfy constraints (e.g. for TOSA)
        # Here we assume vol.shape and flow.shape have known rank and channel dimensions.
        vol = tf.ensure_shape(vol, [None, vol.shape[1], vol.shape[2], vol.shape[3], vol.shape[4]])
        flow = tf.ensure_shape(flow, [None, flow.shape[1], flow.shape[2], flow.shape[3], 3])

        # Flatten batch and depth dims to do a single 2D dense warp call for each depth slice in batch:
        shape = tf.shape(vol)
        B, D, H, W = shape[0], shape[1], shape[2], shape[3]
        C = vol.shape[4]  # channel dimension is static

        vol_flat = tf.reshape(vol, tf.stack([B * D, H, W, C]))
        flow_flat = tf.reshape(flow, tf.stack([B * D, H, W, 3]))

        # Use only first two flow channels (for 2D warp) - ignore Z displacement
        moved_flat = tfa.image.dense_image_warp(vol_flat, flow_flat[..., :2])

        # Reshape back to original 5D shape
        moved = tf.reshape(moved_flat, tf.stack([B, D, H, W, C]))
        return moved

def conv_block(x, filters, convs=2, kernel_size=3, activation='relu'):
    """
    Helper function: stack of convolutional layers with specified filters and activation.
    """
    for _ in range(convs):
        x = layers.Conv3D(filters, kernel_size, padding='same',
                          kernel_initializer='he_normal')(x)
        x = layers.Activation(activation)(x)
    return x

class MyModel(tf.keras.Model):
    """
    Voxelmorph-inspired 3D medical image registration model.
    Takes two inputs: moving and fixed volumes of shape [B, 128, 128, 128, 1].
    Outputs:
      - moved volume (moving warped by predicted flow)
      - flow field (3D displacement field)
    """
    def __init__(self,
                 inshape=(128, 128, 128),
                 enc_features=(16, 32, 32, 32),
                 dec_features=(32, 32, 32, 32, 32, 16, 16),
                 **kwargs):
        super().__init__(**kwargs)
        self.inshape = inshape

        # Inputs will be provided during call.
        self.spatial_transformer = SpatialTransformer(name='moved')

        # Encoder conv blocks
        self.enc_blocks = []
        for f in enc_features:
            block = tf.keras.Sequential([
                layers.Conv3D(f, 3, padding='same', kernel_initializer='he_normal'),
                layers.Activation('relu'),
                layers.Conv3D(f, 3, padding='same', kernel_initializer='he_normal'),
                layers.Activation('relu')
            ])
            self.enc_blocks.append(block)
        self.pool = layers.MaxPool3D(2)

        # Bottom conv block doubles last encoder filters
        self.bottleneck_conv = tf.keras.Sequential([
            layers.Conv3D(enc_features[-1] * 2, 3, padding='same', kernel_initializer='he_normal'),
            layers.Activation('relu'),
            layers.Conv3D(enc_features[-1] * 2, 3, padding='same', kernel_initializer='he_normal'),
            layers.Activation('relu')
        ])

        # Decoder conv blocks
        self.up = layers.UpSampling3D(2)
        self.dec_blocks = []
        for f in dec_features:
            block = tf.keras.Sequential([
                layers.Conv3D(f, 3, padding='same', kernel_initializer='he_normal'),
                layers.Activation('relu'),
                layers.Conv3D(f, 3, padding='same', kernel_initializer='he_normal'),
                layers.Activation('relu')
            ])
            self.dec_blocks.append(block)

        # Final flow output conv layer (3 channels for displacement)
        self.flow_conv = layers.Conv3D(3, 3, padding='same', name='flow')

    def call(self, inputs, training=False):
        # Unpack inputs: moving and fixed volumes
        moving, fixed = inputs

        # Concatenate along channel axis: shape [B, H, W, D, 2]
        x = tf.concat([moving, fixed], axis=-1)

        skips = []
        for enc_block in self.enc_blocks:
            x = enc_block(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck_conv(x)

        # Decoder with skip connections
        for dec_block, skip in zip(self.dec_blocks, reversed(skips)):
            x = self.up(x)
            # If needed, crop skip or x spatial dims to match before concat.
            # Assume shapes compatible for simplicity.
            x = tf.concat([x, skip], axis=-1)
            x = dec_block(x)

        flow = self.flow_conv(x)  # shape [B,D,H,W,3]

        moved = self.spatial_transformer([moving, flow])

        return moved, flow


def my_model_function():
    """
    Instantiate and return the MyModel with default 128^3 input shape.
    """
    return MyModel()


def GetInput():
    """
    Return a tuple of random float tensors matching the expected model inputs:
    Two inputs ("moving" and "fixed") volumes of shape [B, 128, 128, 128, 1].
    Use batch size 1 for simplicity.
    """
    B = 1
    H, W, D = 128, 128, 128
    # TensorFlow 3D conv layers expect inputs in shape: [B, H, W, D, C]
    # Random uniform floats [0,1)
    moving = tf.random.uniform((B, H, W, D, 1), dtype=tf.float32)
    fixed = tf.random.uniform((B, H, W, D, 1), dtype=tf.float32)
    return (moving, fixed)

