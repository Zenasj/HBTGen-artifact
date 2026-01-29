# tf.random.uniform((B, H, W, C), dtype=tf.float32) ← Assuming standard image input batches, e.g. (batch_size, 224, 224, 3)

import tensorflow as tf
from tensorflow import keras

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # The model is composed of three conceptual blocks based on the shared notebook approach:
        # 1) Preprocessing block
        # 2) Convolutional base (e.g. VGG16 without top)
        # 3) Classification head

        # To emulate the nested model scenario, we create submodels for each block.
        # Using VGG16 without top layer and include preprocessing with keras.applications preprocessing layer.
        self.preprocess = keras.Sequential([
            keras.layers.Resizing(224, 224),  # Make sure input is resized correctly
            keras.layers.Lambda(lambda x: x / 255.0)  # Normalize input to [0,1]
        ])

        # Use VGG16 convolutional base (exclude top layer)
        base_model = keras.applications.VGG16(include_top=False, weights='imagenet')
        self.conv_base = keras.Model(inputs=base_model.input, outputs=base_model.get_layer('block5_conv3').output)

        # Classification head: flatten + dense layers matching standard VGG16 classifier with some simplification
        self.classifier = keras.Sequential([
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dense(10, activation='softmax')  # Assuming 10 classes as a simple example
        ])

    def call(self, inputs, training=False):
        # The forward pass follows the separated blocks:
        x = self.preprocess(inputs)
        x = self.conv_base(x)
        x = self.classifier(x)
        return x

def my_model_function():
    """Return an instance of MyModel with weights initialized.
    Weights for VGG16 conv base are loaded from ImageNet; classifier head is randomly initialized."""
    return MyModel()

def GetInput():
    """Generate a random input tensor matching the model input shape:
    Batch size 1, image size 224x224, 3 channels (RGB), dtype float32"""
    return tf.random.uniform((1, 224, 224, 3), minval=0, maxval=255, dtype=tf.float32)

