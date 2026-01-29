# tf.random.uniform((B, 110, 110, 3), dtype=tf.float32) ← Input shape inferred from issue

import tensorflow as tf
from tensorflow.keras.applications import DenseNet121, ResNet50V2
from tensorflow.keras.layers import Flatten

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Initialize both backbone models without top layers, ImageNet weights
        self.densenet = DenseNet121(weights='imagenet', include_top=False, input_shape=(110, 110, 3))
        self.resnet = ResNet50V2(weights='imagenet', include_top=False, input_shape=(110, 110, 3))
        # Flatten layers for each backbone output
        self.flatten = Flatten()

    def call(self, inputs, training=False):
        # Forward pass through DenseNet121 base model
        densenet_features = self.densenet(inputs, training=training)
        # Observed issue: Flatten on DenseNet output throws error because output shape likely contains None dimension in batch,
        # but in functional use, it should not. We add defensive check / ensure call is consistent.

        # Forward pass through ResNet50V2 base model
        resnet_features = self.resnet(inputs, training=training)

        # Flatten outputs
        densenet_flat = self.flatten(densenet_features)
        resnet_flat = self.flatten(resnet_features)

        # Compare flattened outputs numerically with a tolerance since architectures differ.
        # We create a boolean tensor indicating element-wise closeness within tolerance.
        is_close = tf.reduce_all(tf.abs(densenet_flat - resnet_flat) < 1e-5)

        # Return a dict of outputs including comparison result
        # For demonstration, returning a dict as output (Keras allows multiple return types)
        return {
            "densenet_flattened": densenet_flat,
            "resnet_flattened": resnet_flat,
            "are_outputs_close": is_close,
        }

def my_model_function():
    # Instantiate and return the MyModel instance
    return MyModel()

def GetInput():
    # Generate a random input tensor matching shape: batch size 1 (arbitrary), height 110, width 110, channels 3
    return tf.random.uniform((1, 110, 110, 3), dtype=tf.float32)

