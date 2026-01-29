# tf.random.uniform((B, H, W, C), dtype=tf.float32) ← Input is a batch of images, e.g., (batch_size, height, width, 3)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Load pretrained InceptionV3 without top classification layer
        base_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
        self.image_features_extract_model = tf.keras.Model(
            base_model.input, base_model.layers[-1].output
        )
    
    @tf.function
    def call(self, x):
        """
        Forward pass:
        - Extract image features using the pretrained InceptionV3 truncated model.
        - Reshape the output from (B, h, w, c) → (B, h*w, c).
        """
        features = self.image_features_extract_model(x)
        # features shape: (batch_size, height_feature, width_feature, channels)
        batch_size = tf.shape(features)[0]
        features = tf.reshape(features, (batch_size, -1, features.shape[3]))
        return features

def my_model_function():
    # Return an instance of MyModel with pretrained InceptionV3 encoder loaded
    return MyModel()

def GetInput():
    # Return a batch of random images in expected input shape for InceptionV3 base model
    # According to InceptionV3 standard input size: 299x299 RGB images
    batch_size = 16
    height = 299
    width = 299
    channels = 3
    # Random float32 images scaled [0,1) as typical inputs before preprocessing
    return tf.random.uniform((batch_size, height, width, channels), dtype=tf.float32)

