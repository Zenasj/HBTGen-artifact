# tf.random.uniform((2, 3, 3), dtype=tf.uint16) ← Inferred from the test image_shape=(2,3,3) and target dtype uint16

import tensorflow as tf
import numpy as np

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # This model simulates the decoding of a 16-bit PNG image from serialized tf.Example bytes.
        # It encapsulates the decoding logic similar to the TFExampleDecoder with uint16 support,
        # handling the dtype conflict in tf.cond branches by enforcing consistent output types.
        # For this example, we implement a minimal decoder of PNG images from raw serialized bytes 
        # expecting uint16 outputs. In practice, the decoding logic is more complex and involves parsing 
        # the TF Example proto, but here we model the core decode_png and dtype logic relevant to the bug report.

    def call(self, serialized_example):
        """
        Args:
            serialized_example: A scalar tf.string tensor with serialized TF Example containing
                                an 'image/encoded' feature with PNG encoded image bytes.
                                
        Returns:
            Decoded image tensor of dtype tf.uint16 and shape (2, 3, 3) as per the example.
        """

        # Parsing features from serialized TF Example:
        features = {
            'image/encoded': tf.io.FixedLenFeature([], tf.string, default_value=''),
            'image/format': tf.io.FixedLenFeature([], tf.string, default_value='png')
        }
        parsed = tf.io.parse_single_example(serialized_example, features)

        image_buffer = parsed['image/encoded']
        image_format = parsed['image/format']

        # Conditional decode based on image format - simplified for 'png' only.
        def decode_png():
            # decode_png supports uint16 output (channels preserved)
            return tf.io.decode_png(image_buffer, dtype=tf.uint16)

        def decode_jpeg():
            # decode_jpeg always outputs uint8
            return tf.io.decode_jpeg(image_buffer, channels=3)

        # Control flow to select decode function, forcing consistent output dtype.
        # The bug referenced arises if branches output different dtypes: uint16 vs uint8.
        # To fix it, we cast branches to a common dtype (here uint16).
        # For 'png' output, return as uint16. For 'jpeg' fallback, cast uint8->uint16 to unify.
        decoded_image = tf.cond(
            tf.equal(tf.strings.lower(image_format), 'png'),
            true_fn=decode_png,
            false_fn=lambda: tf.cast(decode_jpeg(), tf.uint16)
        )

        return decoded_image

def my_model_function():
    # Simply return an instance of MyModel.
    return MyModel()

def GetInput():
    # Generate a serialized TF Example containing a 2x3x3 uint16 PNG encoded image, matching the example.

    # Construct a uint16 numpy image with shape (2, 3, 3)
    image_shape = (2, 3, 3)
    num_pixels = np.prod(image_shape).item()
    # Create sample image data ranging from 0 to num_pixels-1 as uint16
    image = np.linspace(0, num_pixels - 1, num=num_pixels, dtype=np.uint16).reshape(image_shape)

    # Encode the image as a PNG byte string (uint16 PNG)
    # tf.io.encode_png only supports uint8 and uint16 images.
    # Using TF ops to encode image. We need to run a TF graph to encode.
    # So we use a tf.function for encoding to avoid session/eval.
    @tf.function
    def encode_png(image_uint16):
        return tf.io.encode_png(image_uint16)

    # Run encoding inline (eager mode)
    encoded_png = encode_png(tf.constant(image))

    # Create a serialized tf.train.Example containing 'image/encoded' and 'image/format' features.
    feature = {
        'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[encoded_png.numpy()])),
        'image/format': tf.train.Feature(bytes_list=tf.train.BytesList(value=[b'png'])),
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    serialized_example = example_proto.SerializeToString()

    return tf.constant(serialized_example, dtype=tf.string)

