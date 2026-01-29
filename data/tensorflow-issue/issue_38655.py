# tf.random.uniform((1024, 224, 224, 3), dtype=tf.float32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # As per the reported Keras model: a GlobalMaxPool2D followed by Dense(1000, softmax)
        self.pool = tf.keras.layers.GlobalMaxPool2D(input_shape=(224, 224, 3))
        self.classifier = tf.keras.layers.Dense(1000, activation="softmax")

    def call(self, inputs, training=False):
        x = self.pool(inputs)
        return self.classifier(x)


def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def _decode_and_center_crop(image_bytes):
    """Crops to center of image with padding then scales image_size (224x224)."""
    shape = tf.image.extract_jpeg_shape(image_bytes)
    image_height = shape[0]
    image_width = shape[1]
    image_size = 224

    padded_center_crop_size = tf.cast(
        (image_size / (image_size + 32)) * tf.cast(tf.minimum(image_height, image_width), tf.float32),
        tf.int32,
    )

    offset_height = ((image_height - padded_center_crop_size) + 1) // 2
    offset_width = ((image_width - padded_center_crop_size) + 1) // 2
    crop_window = tf.stack(
        [offset_height, offset_width, padded_center_crop_size, padded_center_crop_size]
    )
    image = tf.image.decode_and_crop_jpeg(image_bytes, crop_window, channels=3)
    return tf.image.resize(image, [image_size, image_size], method="bicubic")

def GetInput():
    # Generate a random input tensor simulating a batch of 1024 images
    # Shape matches model input: (batch_size=1024, height=224, width=224, channels=3)
    # Values are floats to represent preprocessed image data (scaled 0-255 float)
    batch_size = 1024
    height = 224
    width = 224
    channels = 3
    # Using uniform distribution 0-255 to simulate cropped & resized image pixel values
    input_tensor = tf.random.uniform(
        (batch_size, height, width, channels),
        minval=0,
        maxval=255,
        dtype=tf.float32
    )
    return input_tensor

