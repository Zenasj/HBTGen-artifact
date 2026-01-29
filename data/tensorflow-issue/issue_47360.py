# tf.random.uniform((B, img_height, img_width, 3), dtype=tf.float32) ← input shape inferred from data_augmentation input_shape=(img_height, img_width, 3)

import tensorflow as tf
from tensorflow.keras import layers

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define data augmentation pipeline as per the given snippet
        # Using tf.keras.Sequential of preprocessing layers for augmentation
        # These layers apply random horizontal flip, random rotation, and random zoom
        # Input shape is (img_height, img_width, 3), with 3 color channels (RGB)
        self.data_augmentation = tf.keras.Sequential([
            layers.experimental.preprocessing.RandomFlip("horizontal"),
            layers.experimental.preprocessing.RandomRotation(0.1),
            layers.experimental.preprocessing.RandomZoom(0.1),
        ])

    def call(self, inputs, training=False):
        # The augmentation layers are designed to behave differently during training
        # We forward the 'training' flag so the layers apply augmentation only when training=True
        return self.data_augmentation(inputs, training=training)

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Generate a random tensor with shape (batch_size, img_height, img_width, 3)
    # Since the original snippet did not specify batch size or image dimensions explicitly,
    # assume batch size 4, and standard image size 180x180 (commonly used in TF tutorial)
    batch_size = 4
    img_height = 180
    img_width = 180
    channels = 3
    # Use floating point values in [0,1) as pixel inputs, dtype float32
    return tf.random.uniform((batch_size, img_height, img_width, channels), dtype=tf.float32)

