# tf.random.uniform((B, H, W, C), dtype=tf.float32)  ← Assume typical image input (batch, height, width, channels) with float32

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Model to simulate the behavior of TF 2.3 data augmentation pipeline
        # Using tf.keras.Sequential with RandomFlip and RandomRotation 90 degrees (in radians)
        # Note: rotation factor is in radians, RandomRotation rotates by factor * 2*pi.
        # 90 degrees = pi/2 radians, so factor = 90/360=0.25
        self.aug_tf23 = tf.keras.Sequential([
            tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
            tf.keras.layers.experimental.preprocessing.RandomRotation(0.25),  # 90 degrees rotation
        ])
        # Model to simulate fixed TF 2.4+ behavior (no black lines artifact)
        # For demonstration, we replicate the same augmentation but with correct fill_mode to avoid black borders.
        # By default, RandomRotation uses 'constant' fill_mode with value=0, causing black lines.
        # Fix: use fill_mode='reflect' or 'nearest' to avoid black edges.
        self.aug_tf24 = tf.keras.Sequential([
            tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
            tf.keras.layers.experimental.preprocessing.RandomRotation(
                0.25,
                fill_mode='reflect'  # reflect fill prevents black lines artifacts
            ),
        ])

    def call(self, inputs, training=None):
        # During training, apply augmentations.
        # Simulate TF 2.3 output with artifact and TF 2.4 fix output.
        # For the purpose of comparison, return boolean tensor whether outputs differ significantly.
        # Note: In actual usage, only one pipeline would be used.
        tf23_out = self.aug_tf23(inputs, training=training)
        tf24_out = self.aug_tf24(inputs, training=training)

        # Compute absolute difference between the two augmented images
        diff = tf.abs(tf23_out - tf24_out)
        # Threshold the difference to detect pixels likely affected by black line artifact
        # Using a small tolerance, e.g., 0.05 (assuming normalized images in [0,1])
        diff_mask = tf.reduce_any(diff > 0.05, axis=-1)  # shape (B, H, W)

        # Return:
        #   A dictionary including both augmented outputs and a mask showing difference presence
        # This allows downstream inspection/comparison
        return {
            'tf23_augmented': tf23_out,
            'tf24_fixed_augmented': tf24_out,
            'difference_mask': diff_mask,
        }

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random float32 tensor with shape typical for image input batches.
    # Assume batch size 4, image size 128x128, RGB channels (3)
    # Inputs normalized between 0 and 1
    batch_size = 4
    height = 128
    width = 128
    channels = 3
    input_tensor = tf.random.uniform(
        shape=(batch_size, height, width, channels),
        minval=0.0, maxval=1.0,
        dtype=tf.float32
    )
    return input_tensor

