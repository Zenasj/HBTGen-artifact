# tf.random.uniform((B,)) with dtype=string representing encoded images

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Load a pretrained ResNet101 model expecting 224x224 RGB float32 inputs
        self.model = tf.keras.applications.ResNet101(
            include_top=True,
            weights="imagenet",
            input_shape=(224, 224, 3),
            classes=1000,
        )

    def call(self, x):
        # Preprocess: decode image bytes, disable expand_animations to get a rank 3 tensor,
        # resize to 224x224, then call ResNet101.
        x = self.preprocess(x)
        return self.model(x)

    def preprocess(self, inputs):
        # Map decoding over the batch, decoding JPEG/PNG bytes to float32 tensors in [0,1]
        # This avoids the ValueError related to shape and string tensor broadcast.
        x = tf.map_fn(
            self.decode_bytes, 
            inputs, 
            fn_output_signature=tf.float32  # Updated per TF 2.8 deprecation notes
        )
        # Now resize all images to 224 x 224 as expected by ResNet.
        x = tf.image.resize(x, (224, 224))
        return x

    def decode_bytes(self, x):
        # Decode each image byte tensor with expand_animations=False to avoid rank issues
        # and produce a tensor of shape (H,W,3) with float32 values.
        return tf.io.decode_image(
            x, 
            channels=3, 
            dtype=tf.float32, 
            expand_animations=False
        )

def my_model_function():
    # Return an instance of MyModel with pretrained weights loaded
    return MyModel()

def GetInput():
    # Generate a random 3-channel 330x330 image, encode to PNG bytes,
    # wrap in a batch dimension as a string tensor.
    # This matches input expected by MyModel: batch of image byte strings.
    import numpy as np
    from PIL import Image
    import io

    img_array = (np.random.random((330, 330, 3)) * 255).astype(np.uint8)
    pil_img = Image.fromarray(img_array, 'RGB')
    buffer = io.BytesIO()
    pil_img.save(buffer, format='PNG')
    img_bytes = buffer.getvalue()

    # Encode as scalar string tensor batch of size 1
    input_tensor = tf.constant([img_bytes])
    return input_tensor

