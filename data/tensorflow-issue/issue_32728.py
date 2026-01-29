# tf.random.uniform((None, None, None, 3), dtype=tf.float32)  ← Assumed typical input shape for InceptionV3 is (B, H, W, 3) with float32

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Using the Keras InceptionV3 model without top classification layer, pretrained on ImageNet
        # This assumes input is compatible: float32 tensor with shape (batch, height, width, 3)
        # Using include_top=False to get feature representation only
        self.inception = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')

    def call(self, inputs, training=False):
        # Forward pass through the Inception V3 base
        features = self.inception(inputs, training=training)
        # Global average pooling to get fixed-size output vector per image
        pooled = tf.keras.layers.GlobalAveragePooling2D()(features)
        return pooled

def my_model_function():
    # Return an instance of MyModel with pretrained InceptionV3 base
    return MyModel()

def GetInput():
    # Generate a random batch input tensor to match Inception V3 expected input shape
    # InceptionV3 typically expects 299x299 images with 3 channels, float32 in [0, 255] preprocessed to [-1, 1]
    batch_size = 1  # batch size 1 for simplicity
    height, width, channels = 299, 299, 3
    x = tf.random.uniform((batch_size, height, width, channels), minval=0, maxval=255, dtype=tf.float32)
    # Preprocess input according to InceptionV3 expectations
    x = tf.keras.applications.inception_v3.preprocess_input(x)
    return x

