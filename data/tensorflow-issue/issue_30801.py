# tf.random.uniform((batch_size, 224, 224, 3), dtype=tf.float32)
import tensorflow as tf
import numpy as np

# The original issue describes a custom data generator yielding batches of images
# with shape (batch_size, 224, 224, 3) and labels as one-hot vectors over 11 classes.
# However, the main discussion involves memory leak behavior with multiprocessing
# in Keras fit_generator and data pipelines. There is no direct model code,
# but it references usage of resnet50 preprocessing, so we assume a typical image
# classification model architecture for the input size and label shape.

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Using a simple backbone model with input shape (224, 224, 3)
        # and output classes=11 as per the DataGenerator labels.
        # We use tf.keras.applications.ResNet50 as feature extractor,
        # including preprocessing (the generator already does this),
        # so here we just build from inputs without preprocessing layer.

        # Instantiate the base ResNet50 model without top layers, weights=None for simplicity
        self.backbone = tf.keras.applications.ResNet50(
            include_top=False,
            weights=None,
            input_shape=(224, 224, 3),
            pooling='avg',  # Global average pooling after final conv block
        )
        self.classifier = tf.keras.layers.Dense(11, activation='softmax')

    def call(self, inputs, training=False):
        # inputs shape: (batch_size, 224, 224, 3), preprocessed externally as per resnet50
        x = self.backbone(inputs, training=training)
        x = self.classifier(x)
        return x


def my_model_function():
    # Return an instance of MyModel
    return MyModel()


def GetInput():
    # Return a random tensor input matching expected input to MyModel
    # (batch_size, 224, 224, 3), float32, values roughly in resnet50 preprocessing range
    batch_size = 8  # reasonable batch size for input example
    # ResNet50 preprocessing expects pixels scaled to mode:
    # preprocess_input for resnet50 converts RGB to BGR, zero-centers each color channel.
    # For simplicity, generate random uint8 [0,255] and apply preprocessing here.

    # Create uint8 random images in [0,255]
    x_uint8 = tf.random.uniform((batch_size, 224, 224, 3), minval=0, maxval=256, dtype=tf.int32)
    x_uint8 = tf.cast(x_uint8, tf.uint8)

    # Convert to float32 for preprocessing
    x_float = tf.cast(x_uint8, tf.float32)

    # Apply resnet50 preprocessing (BGR zero-centering)
    x_preprocessed = tf.keras.applications.resnet50.preprocess_input(x_float)

    return x_preprocessed

