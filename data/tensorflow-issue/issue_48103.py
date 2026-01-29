# tf.random.uniform((B, 32, 32, 3), dtype=tf.float32)  ← Assuming batch size B is dynamic, input shape is fixed (32,32,3)

import tensorflow as tf
import efficientnet.keras as efn

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Instantiate the official EfficientNetB0 base from tf.keras.applications
        # Note: input_shape must be (32,32,3), which is smaller than standard ImageNet size (224,224,3).
        # We set include_top=False to exclude the classifier head.
        self.official_base = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights="imagenet",
            input_shape=(32, 32, 3),
            pooling=None,  # We will add pooling layer explicitly
        )
        # Instantiate the non-official EfficientNetB0 base from qubvel's efficientnet.keras
        self.non_official_base = efn.EfficientNetB0(
            include_top=False,
            weights="imagenet",
            input_shape=(32, 32, 3),
            pooling=None,
        )
        # Shared global average pooling and dense classification head for both models
        self.global_avg_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.classifier = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, inputs, training=False):
        # Forward pass for official model
        x_official = self.official_base(inputs, training=training)
        x_official = self.global_avg_pool(x_official)
        out_official = self.classifier(x_official)

        # Forward pass for non-official model
        x_non_official = self.non_official_base(inputs, training=training)
        x_non_official = self.global_avg_pool(x_non_official)
        out_non_official = self.classifier(x_non_official)

        # Compare predictions between official and non-official models
        # We'll compute the absolute difference between outputs.
        diff = tf.abs(out_official - out_non_official)

        # Return a dictionary-like output including:
        # - output of official model
        # - output of non-official model
        # - difference tensor (L1 difference)
        # This lets caller inspect predictions and differences for reproducibility analysis.
        return {
            "official": out_official,
            "non_official": out_non_official,
            "difference": diff
        }

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a batch of random images with shape (batch_size, 32, 32, 3)
    # float32 values in [0, 1], matching preprocessing assumed by EfficientNet weights
    batch_size = 4  # Example batch size; can be any positive int
    return tf.random.uniform((batch_size, 32, 32, 3), dtype=tf.float32)

