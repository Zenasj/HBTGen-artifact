# tf.random.uniform((batch, 1), dtype=tf.string) ← Input is batch of single-element string tensors (JPEG-encoded images)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
    
    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 1], dtype=tf.string)])
    def call(self, inputs):
        # inputs shape: [batch, 1], dtype=tf.string
        # Each element is a single JPEG-encoded image string in a 1-element tensor.

        def pre_input(image):
            # image: shape [1], type tf.string
            # Extract scalar string from the 1-element tensor
            image = image[0]
            # Decode image to uint8 tensor [height, width, channels], channels inferred automatically
            image = tf.io.decode_image(image, channels=3, expand_animations=False)
            image = tf.cast(image, dtype=tf.float32)
            # Normalize image to roughly [-1, 1]
            image = (image - 127.0) / 128.0
            return image
        
        # Important: Specify fn_output_signature with shape/dtype for TF Lite compatibility
        # Here we must specify the output shape of each decoded image.
        # Since decode_image output shape is dynamic, we assume fixed size images for simplicity.
        # Let's assume input images are 64x64x3 (height x width x channel).
        # This assumption matches the original example context.

        fn_output_signature = tf.TensorSpec(shape=(64, 64, 3), dtype=tf.float32)
        images = tf.map_fn(pre_input, inputs, fn_output_signature=fn_output_signature)
        # images shape: [batch, 64, 64, 3]
        
        return images


def my_model_function():
    # Return an instance of the model
    return MyModel()


def GetInput():
    # Generate a batch of JPEG-encoded 64x64x3 images wrapped as [batch,1] string tensors.

    batch = 4  # Choose a small batch size for example

    # Create random uint8 images with shape [batch, 64, 64, 3]
    random_imgs = tf.random.uniform(
        shape=(batch, 64, 64, 3), minval=0, maxval=255, dtype=tf.int32)
    random_imgs = tf.cast(random_imgs, dtype=tf.uint8)

    # Encode each image to JPEG
    def encode_img(img):
        return tf.io.encode_jpeg(img)
    encoded_imgs = tf.map_fn(encode_img, random_imgs, dtype=tf.string)

    # Reshape encoded images to [batch, 1]
    encoded_imgs = tf.reshape(encoded_imgs, [batch, 1])

    return encoded_imgs

