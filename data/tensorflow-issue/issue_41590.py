# tf.random.uniform((8, 256, 256, 3), dtype=tf.float32)
import tensorflow as tf
import numpy as np

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Simple classification model matching the issue example
        self.flatten = tf.keras.layers.Flatten(input_shape=(256, 256, 3))
        self.dense1 = tf.keras.layers.Dense(100, activation='relu')
        self.dense2 = tf.keras.layers.Dense(10)

    def call(self, inputs, training=False):
        x = self.flatten(inputs)
        x = self.dense1(x)
        return self.dense2(x)


def my_model_function():
    # Return an instance of MyModel with the same architecture and weights initialized normally
    return MyModel()


def GetInput():
    # Generate a valid input tensor matching the model's expected input: batch of 8 images
    # Data is float32 in range [-0.5, 0.5] as normalized by (jpeg_decode / 255 - 0.5)
    # We mimic the data pipeline by:
    # 1) Create uint8 zeros image batch (8, 256, 256, 3)
    # 2) Encode to jpeg and decode with explicit shape (to avoid TPU shape inference error)
    # 3) Normalize as float32 in [-0.5, 0.5]
    # For simplicity in this standalone function, we directly produce float32
    batch_size = 8
    height = 256
    width = 256
    channels = 3
    # Use uniform random noise instead of zeros to simulate realistic input
    uint8_images = tf.cast(
        tf.random.uniform((batch_size, height, width, channels), minval=0, maxval=256, dtype=tf.int32),
        tf.uint8
    )
    def encode_jpg(image):
        return tf.io.encode_jpeg(image, quality=95, optimize_size=True, chroma_downsampling=False)

    def decode_jpg(jpeg_bytes):
        # Decode with explicit shape to satisfy TPU static shape requirements
        decoded = tf.image.decode_jpeg(jpeg_bytes, channels=3)
        decoded = tf.reshape(decoded, [height, width, channels])
        return decoded

    # Encode and decode each image individually via tf.map_fn
    # This simulates realistic jpeg encode/decode pipeline with static shapes
    jpeg_images = tf.map_fn(encode_jpg, uint8_images, fn_output_signature=tf.string)
    decoded_images = tf.map_fn(decode_jpg, jpeg_images, fn_output_signature=tf.uint8)

    # Normalize to float32 in [-0.5, 0.5]
    float_images = tf.cast(decoded_images, tf.float32) / 255.0 - 0.5

    # Return as batch tensor directly (model expects tensor only, no label here)
    return float_images

