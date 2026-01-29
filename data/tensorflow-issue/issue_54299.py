# tf.random.uniform((1024*16, 224, 224, 3), dtype=tf.float32) ← Input shape inferred from example batch size * image size with 3 channels

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define a simple model similar to the reported example
        self.global_max_pool = tf.keras.layers.GlobalMaxPool2D()
        self.classifier = tf.keras.layers.Dense(1000, activation="softmax")

    def call(self, inputs, training=False):
        x = self.global_max_pool(inputs)
        return self.classifier(x)

def _decode_and_center_crop(image_bytes):
    """Crops center of JPEG image with padding then resizes to 224x224."""
    shape = tf.image.extract_jpeg_shape(image_bytes)
    image_height, image_width = shape[0], shape[1]
    image_size = 224

    padded_center_crop_size = tf.cast(
        (image_size / (image_size + 32))
        * tf.cast(tf.minimum(image_height, image_width), tf.float32),
        tf.int32,
    )

    offset_height = ((image_height - padded_center_crop_size) + 1) // 2
    offset_width = ((image_width - padded_center_crop_size) + 1) // 2
    crop_window = tf.stack(
        [offset_height, offset_width, padded_center_crop_size, padded_center_crop_size]
    )
    image = tf.image.decode_and_crop_jpeg(image_bytes, crop_window, channels=3)
    image = tf.image.resize(image, [image_size, image_size])
    return tf.cast(image, dtype=tf.float32)

def my_model_function():
    # Returns an instance of the Keras model defined above
    return MyModel()

def GetInput():
    # Return a batch of random uint8 image bytes shaped like JPEG encoded images placeheld by random data.
    # Because real JPEG byte encoding is complex, for demonstration we create random image tensors of shape (batch, 224, 224, 3).
    # This matches the expected float32 input after preprocessing.
    # Batch size inferred from example: 1024 is batch size used in original code (but to keep reasonable size here, smaller batch)
    batch_size = 16
    image_size = 224
    # Simulate images as float32 tensors similar to decoded images expected by the model:
    # tf.random.uniform for float32 [0, 255)
    return tf.random.uniform((batch_size, image_size, image_size, 3), minval=0, maxval=255, dtype=tf.float32)

