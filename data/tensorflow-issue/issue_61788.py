# tf.random.uniform((B, 96, 96, 3), dtype=tf.float32) ← Input shape inferred from the cats_and_dogs dataset image size (96,96,3)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Data augmentation layers as used in the original code
        self.data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip('horizontal'),
            tf.keras.layers.RandomRotation(0.2),
        ])

        # Preprocessing for MobileNetV2
        self.preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

        # Base pretrained MobileNetV2 model without the classification head
        IMG_SIZE = (96, 96)
        IMG_SHAPE = IMG_SIZE + (3,)
        self.base_model = tf.keras.applications.MobileNetV2(
            input_shape=IMG_SHAPE,
            include_top=False,
            weights='imagenet'
        )
        self.base_model.trainable = False  # Initially freeze the base

        # Global average pooling for feature extraction
        self.global_average_layer = tf.keras.layers.GlobalAveragePooling2D()

        # Dropout layer for regularization
        self.dropout = tf.keras.layers.Dropout(0.2)

        # Final prediction layer outputting 1 logit (binary classification)
        self.prediction_layer = tf.keras.layers.Dense(1)

    def call(self, inputs, training=False):
        # Apply data augmentation only during training
        x = tf.cond(
            tf.cast(training, tf.bool),
            true_fn=lambda: self.data_augmentation(inputs),
            false_fn=lambda: inputs
        )
        # Preprocess input for MobileNetV2
        x = self.preprocess_input(x)
        # Pass through base model; training=False to keep batchnorm layers in inference mode initially
        x = self.base_model(x, training=False)
        # Pool features
        x = self.global_average_layer(x)
        # Apply dropout only during training
        x = self.dropout(x, training=training)
        # Final predictions (logits)
        x = self.prediction_layer(x)
        return x

    def fine_tune(self, fine_tune_at=100):
        """Unfreeze the base model from layer `fine_tune_at` onward."""
        # Unfreeze entire base model first
        self.base_model.trainable = True
        # Freeze layers before fine_tune_at
        for layer in self.base_model.layers[:fine_tune_at]:
            layer.trainable = False


def my_model_function():
    """
    Return an instance of the MyModel class, which is a MobileNetV2 based transfer learning model
    with data augmentation and fine-tuning capability.
    """
    model = MyModel()
    return model


def GetInput():
    """
    Return a random tensor input matching the expected input shape of MyModel:
    batch size 32, image size (96,96), 3 channels, dtype float32 in [0,255].
    """
    BATCH_SIZE = 32
    IMG_SIZE = (96, 96)
    # Generate random images in [0,255], float32 type
    input_tensor = tf.random.uniform(
        (BATCH_SIZE, IMG_SIZE[0], IMG_SIZE[1], 3),
        minval=0,
        maxval=255,
        dtype=tf.float32
    )
    return input_tensor

