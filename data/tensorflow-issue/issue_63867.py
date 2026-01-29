# tf.random.uniform((B, 128, 128, 3), dtype=tf.float32) ← inferred from data_augmentation input_shape and image size in issue

import tensorflow as tf
from tensorflow.keras import layers

class MyModel(tf.keras.Model):
    def __init__(self, num_classes=525, img_height=128, img_width=128,
                 activation1='swish', activation2='elu'):
        super().__init__()
        # Data augmentation layers as used in original Sequential model from issue
        self.data_augmentation = tf.keras.Sequential([
            layers.RandomFlip("horizontal", input_shape=(img_height, img_width, 3)),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ])

        self.conv_block = tf.keras.Sequential([
            # Following the conv layers from issue's Sequential model:
            layers.Conv2D(32, 3, padding='same', activation=activation1),
            layers.MaxPooling2D(),
            layers.Conv2D(48, 3, padding='same', activation=activation2),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding='same', activation=activation1),
            layers.MaxPooling2D(),
            layers.Dropout(0.15),
            layers.Flatten(),
            layers.Dense(128, activation=activation2),
            layers.Dense(num_classes)
        ])

        # Include normalization layer explicitly to match training normalization
        self.normalization_layer = layers.Rescaling(1./255)

    def call(self, inputs, training=False):
        x = self.data_augmentation(inputs, training=training)
        x = self.normalization_layer(x)
        x = self.conv_block(x, training=training)
        return x

def my_model_function():
    # Instantiate a model matching the reported example with 525 classes and input 128x128x3
    return MyModel(num_classes=525, img_height=128, img_width=128,
                   activation1='swish', activation2='elu')

def GetInput():
    # Return a random tensor that's compatible with MyModel input requirements
    # Batch size chosen as 32 for example, matching typical batch size from the issue
    batch_size = 32
    height = 128
    width = 128
    channels = 3
    return tf.random.uniform((batch_size, height, width, channels), dtype=tf.float32)

