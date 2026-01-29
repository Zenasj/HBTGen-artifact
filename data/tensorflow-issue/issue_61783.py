# tf.random.uniform((B, 224, 224, 3), dtype=tf.float32) ← Assuming input shape from issue: images resized to 224x224 with 3 channels

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, num_classes=8):
        super().__init__()
        # Use MobileNetV3Large minimalistic as a feature extractor (frozen)
        self.base_model = tf.keras.applications.MobileNetV3Large(
            input_shape=(224, 224, 3),
            include_top=False,
            weights='imagenet',
            minimalistic=True  # as used in original code
        )
        self.base_model.trainable = False

        self.global_pool = tf.keras.layers.GlobalMaxPooling2D()
        self.dropout = tf.keras.layers.Dropout(0.2)
        self.classifier = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, inputs, training=False):
        x = self.base_model(inputs, training=False)  # base model is frozen
        x = self.global_pool(x)
        x = self.dropout(x, training=training)
        return self.classifier(x)


def my_model_function():
    # Create and return model instance with 8 classes (default from issue)
    return MyModel(num_classes=8)


def GetInput():
    # Return a random input tensor representing a batch of RGB images 224x224
    # Batch size: for example 16 as mentioned in discussion
    batch_size = 16
    input_tensor = tf.random.uniform(
        shape=(batch_size, 224, 224, 3),
        minval=0, maxval=1,
        dtype=tf.float32
    )
    return input_tensor

