# tf.random.uniform((B, 180, 180, 3), dtype=tf.float32) ← input shape inferred from the example image size and batch example

import tensorflow as tf

class MyModel(tf.keras.Model):
    """
    A model that emulates the data augmentation pipeline described in the issue.

    The main point is to include RandomFlip, RandomRotation, and RandomZoom layers in a Sequential,
    matching typical data augmentation from Keras tutorials.

    Known issue from the original discussion:
    - RandomRotation and RandomZoom may throw errors on Apple M1 GPU due to missing kernel support.
      The problem arises from stateful random ops for GPU devices on M1.
    - To avoid the runtime error, these layers might be disabled or run only on CPU.

    This model encapsulates the augmentation layers in a tf.keras.Sequential and applies them.
    """

    def __init__(self):
        super().__init__()
        # Input shape is (180, 180, 3) based on user's image_size from the example.

        # Data augmentation pipeline:
        # We keep RandomFlip (horizontal) always.
        # The RandomRotation and RandomZoom layers are included logically here,
        # but in practice may cause runtime error on M1 GPU with TF 2.5 due to unsupported op kernel.
        #
        # The forward method will attempt to run all, but users may want to override to exclude
        # or fallback if running on M1 GPU.

        self.data_augmentation = tf.keras.Sequential([
            tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
            tf.keras.layers.experimental.preprocessing.RandomRotation(0.1),
            tf.keras.layers.experimental.preprocessing.RandomZoom(0.1),
        ])

    def call(self, inputs, training=False):
        """
        Apply data augmentation only in training mode.

        Warning:
        On Apple M1 GPU with TF 2.5, RandomRotation and RandomZoom may trigger a NotFoundError
        for 'RngReadAndSkip' op kernel missing on GPU devices.

        As a workaround, users can run with training=False or remove these layers from the pipeline.
        """
        return self.data_augmentation(inputs, training=training)


def my_model_function():
    """
    Returns an instance of MyModel.
    """
    return MyModel()


def GetInput():
    """
    Generate a random tensor input matching the input expected by MyModel.

    From the issue:
    - Input images are 180 x 180 with 3 channels (RGB).
    - Batch size used in example is 32.
    """
    batch_size = 32
    height = 180
    width = 180
    channels = 3

    # Generate random float32 tensor in [0, 255] range to simulate image pixels,
    # since augmentation layers expect pixel values and apply transformations accordingly.
    # dtype matches typical image datatype before preprocessing.
    return tf.random.uniform(
        shape=(batch_size, height, width, channels),
        minval=0,
        maxval=255,
        dtype=tf.float32
    )

