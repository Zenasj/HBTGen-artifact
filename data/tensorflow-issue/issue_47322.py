# tf.random.uniform((None, None, 3), dtype=tf.float32) ← The input expected is a single 3-channel image of arbitrary height and width

import tensorflow as tf
import tensorflow_hub as hub

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Load the DELF model from TensorFlow Hub as a KerasLayer
        # Note: The original code uses hub.Module which is TF1 style.
        # Here we adapt to TF2 by using hub.KerasLayer.
        # The DELF module expects inputs of shape [H, W, 3], batch size 1 is typically required.
        self.delf_layer = hub.KerasLayer(
            "https://tfhub.dev/google/delf/1",
            output_key="descriptors",
            trainable=False
        )
        self.locations_layer = hub.KerasLayer(
            "https://tfhub.dev/google/delf/1",
            output_key="locations",
            trainable=False
        )

    @tf.function(jit_compile=True)
    def call(self, inputs):
        # inputs: a 3D tensor [H, W, 3], float32 representing a single image, pixel values [0,1].
        # DELF model from TF Hub expects batch dimension for TF2, so expand dims.
        img_batch = tf.expand_dims(inputs, axis=0)
        
        # DELF TF hub module input dict for TF2 KerasLayer requires a dict
        inputs_dict = {
            'image': tf.cast(img_batch, tf.float32),
            'score_threshold': 100.0,
            'image_scales': tf.constant([0.25, 0.3536, 0.5, 0.7071, 1.0, 1.4142, 2.0], dtype=tf.float32),
            'max_feature_num': tf.constant(1000, dtype=tf.int32),
        }
        
        # Invoke descriptors and locations from the module. 
        # Since hub.KerasLayer wraps the module, assume single call returns a dict with keys
        # Note: TF Hub KerasLayer usage for a TF v1 hub.Module is limited, so emulate it.
        desc = self.delf_layer(inputs_dict)  # descriptors tensor
        loc = self.locations_layer(inputs_dict)  # locations tensor
        
        # Remove batch dimension for output for ease of use
        desc = tf.squeeze(desc, axis=0)
        loc = tf.squeeze(loc, axis=0)
        
        return {'locations': loc, 'descriptors': desc}

def my_model_function():
    # Return an instance of MyModel; weights are loaded from TF Hub automatically
    return MyModel()

def GetInput():
    # Create a random image tensor with 3 channels and arbitrary height and width
    # DELF commonly uses natural image input shapes. Let's pick 224x224 for testing.
    height, width = 224, 224
    return tf.random.uniform((height, width, 3), dtype=tf.float32)

