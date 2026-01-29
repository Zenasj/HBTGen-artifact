# tf.random.uniform((64, 1), dtype=tf.string) ← Input is a batch of 1-D string tensors, shape [batch_size, 1] containing encoded image bytes

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # No additional layers, just decoding image bytes to tensor using tf.io.decode_image
        # Note: tf.io.decode_image supports jpeg, png, gif decoding but relies on ops not fully supported in TFLite without flex delegate.
        # Here we implement the minimal decoding model.

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 1], dtype=tf.string)])
    def call(self, inputs):
        # inputs shape: [batch_size, 1], where inputs[i][0] is the encoded image bytes string tensor
        # For demonstration, decode the first example in the batch
        # NOTE: tf.io.decode_image returns [height, width, channels], dynamic size depending on image.
        # For tf.function with static shape, real usage may require fixed-size images or padding.
        # We decode each element independently in a map for batch decode.

        # Decode all inputs string tensors into images
        def decode_fn(x):
            return tf.io.decode_image(x[0], channels=3, dtype=tf.uint8)
        
        decoded = tf.map_fn(decode_fn, inputs, fn_output_signature=tf.TensorSpec(shape=(None, None, 3), dtype=tf.uint8))
        # The output is a Ragged Tensor-like collection because images can have different sizes.
        # But tf.map_fn returns a tf.Tensor if output shapes are consistent, else a nested structure.
        # For simplicity, return the decoded tensor directly.
        return decoded


def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Create a random 64x64 RGB image with uint8 pixels
    image = tf.random.uniform(shape=[64,64,3], minval=0, maxval=256, dtype=tf.int32)
    image_uint8 = tf.cast(image, tf.uint8)

    # Encode the image to JPEG bytes
    jpeg_bytes = tf.io.encode_jpeg(image_uint8)

    # Prepare input shape [batch_size, 1], i.e. batch with 1 string tensor per batch element
    # We'll stack the same image twice as an example batch input size 2
    input_batch = tf.stack([jpeg_bytes, jpeg_bytes])
    input_batch = tf.reshape(input_batch, [-1, 1])  # shape (2,1)

    return input_batch

