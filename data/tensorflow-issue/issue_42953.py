# tf.random.uniform((B, 224, 224, 3), dtype=tf.float32) ← Assuming input batch size B is dynamic, 3 channel color images 224x224

import tensorflow as tf
from tensorflow import keras


class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Tiny Darknet-like architecture as defined in the shared issue
        
        self.model = keras.Sequential([
            keras.layers.Conv2D(16, (3, 3), strides=1, padding="same", input_shape=(224,224,3)),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(alpha=0.1),
            keras.layers.MaxPooling2D((2, 2), strides=2),

            keras.layers.Conv2D(32, (3, 3), strides=1, padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(alpha=0.1),
            keras.layers.MaxPooling2D((2, 2), strides=2),

            keras.layers.Conv2D(16, (1, 1), strides=1, padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(alpha=0.1),

            keras.layers.Conv2D(128, (3, 3), strides=1, padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(alpha=0.1),

            keras.layers.Conv2D(16, (1, 1), strides=1, padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(alpha=0.1),

            keras.layers.Conv2D(128, (3, 3), strides=1, padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(alpha=0.1),
            keras.layers.MaxPooling2D((2, 2), strides=2),

            keras.layers.Conv2D(32, (1, 1), strides=1, padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(alpha=0.1),

            keras.layers.Conv2D(256, (3, 3), strides=1, padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(alpha=0.1),

            keras.layers.Conv2D(32, (1, 1), strides=1, padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(alpha=0.1),

            keras.layers.Conv2D(256, (3, 3), strides=1, padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(alpha=0.1),
            keras.layers.MaxPooling2D((2, 2), strides=2),

            keras.layers.Conv2D(64, (1, 1), strides=1, padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(alpha=0.1),

            keras.layers.Conv2D(512, (3, 3), strides=1, padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(alpha=0.1),

            keras.layers.Conv2D(64, (1, 1), strides=1, padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(alpha=0.1),

            keras.layers.Conv2D(512, (3, 3), strides=1, padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(alpha=0.1),

            keras.layers.Conv2D(128, (1, 1), strides=1, padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(alpha=0.1),

            keras.layers.Conv2D(1000, (1, 1)),
            keras.layers.BatchNormalization(),

            keras.layers.AveragePooling2D(),
            keras.layers.Flatten(),
            keras.layers.Dense(1, activation="sigmoid")
        ])

    def call(self, inputs, training=False):
        return self.model(inputs, training=training)


def my_model_function():
    # Return an instance of MyModel with default initialization
    model = MyModel()
    # Compile similarly as original code (optimizer Adam and binary crossentropy from logits True)
    # Note: The original model uses sigmoid activation for output and from_logits=True in loss,
    # which is slightly inconsistent but we'll keep activation sigmoid and from_logits=True per code.
    # (Usually for sigmoid output, from_logits=False. Could adjust.)
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=["accuracy"]
    )
    return model


def GetInput():
    # Return a random tensor input matching input expected by the model: batch size 1 assumed for this sample
    # Shape is (batch_size, 224, 224, 3), dtype float32 typical for image inputs
    return tf.random.uniform((1, 224, 224, 3), dtype=tf.float32)

