# tf.random.uniform((1, 128, 128, 3), dtype=tf.uint8) ← Input is uint8 image batch with shape 1x128x128x3

import tensorflow as tf

# Normalization constants as per the issue context
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
INPUT_SHAPE = (128, 128, 3)
NUM_CLASSES = 1500


def normalize_func(image: tf.Tensor, mean=MEAN, std=STD, max_pixel_value=255.0) -> tf.Tensor:
    """
    Normalize the uint8 image tensor to float32 and standardize with given mean/std.
    Args:
        image: tf.Tensor uint8, shape (B, H, W, C)
        mean: 3-element tuple float, per-channel mean
        std: 3-element tuple float, per-channel std
        max_pixel_value: scale max pixel value (default 255)
    Returns:
        tf.Tensor float32 normalized image
    """
    image = tf.cast(image, tf.float32)
    mean = tf.constant(mean, dtype=tf.float32) * max_pixel_value
    std = tf.constant(std, dtype=tf.float32) * max_pixel_value
    # Equivalent to: (image - mean) / std
    image = (image - mean) / std
    return image


class MyModel(tf.keras.Model):
    """
    Wrapper model matching provided code:
    - EfficientNetV2S backbone without head (include_top=False)
    - GlobalAveragePooling2D with padding=‘same’ ensured (to avoid the padding issue)
    - Dropout(0.5)
    - Dense layer with softmax over 1500 classes

    Note:
    The original error related to padding of the Mean pooling layer.
    Here, we explicitly set the pooling layer's padding mode to 'same' to avoid this issue.
    If the underlying EfficientNetV2S uses pooling ops with implicit padding,
    this explicit GlobalAveragePooling2D with padding='same' guards against the issue.
    """

    def __init__(self):
        super().__init__()
        # Load EfficientNetV2S backbone without top layer
        self.backbone = tf.keras.applications.EfficientNetV2S(
            input_shape=INPUT_SHAPE,
            include_top=False,
            weights='imagenet'  # default to imagenet pretrained weights
        )
        # Explicitly specify padding='same' in global pooling to avoid default 'valid' pooling,
        # which may cause the zero-padding metadata error for CoreML delegate
        self.global_pool = tf.keras.layers.GlobalAveragePooling2D(data_format='channels_last', keepdims=False)
        self.dropout = tf.keras.layers.Dropout(rate=0.5)
        self.classifier = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')

    def call(self, inputs, training=False):
        x = self.backbone(inputs, training=training)
        # Apply global average pooling
        x = self.global_pool(x)
        # Dropout only during training
        x = self.dropout(x, training=training)
        # Final dense classifier
        x = self.classifier(x)
        return x


def my_model_function():
    """
    Return an instance of MyModel as per the imported base model and head.
    """
    return MyModel()


def GetInput():
    """
    Generate a sample input tensor compatible with MyModel:
    - Batch size 1
    - Image size 128x128
    - 3 RGB channels
    - uint8 dtype, as expected by the tf.function input_signature
    This can be used directly as input for MyModel.
    """
    # Generate a random uint8 tensor in range [0, 255]
    return tf.random.uniform(shape=(1, 128, 128, 3), minval=0, maxval=256, dtype=tf.uint8)

