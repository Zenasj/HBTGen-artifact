# tf.random.uniform((B, 150, 150, 3), dtype=tf.float32)  # Input shape based on the cats_vs_dogs dataset example

import tensorflow as tf
import numpy as np

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Data augmentation layers as defined in the issue's code
        self.data_augmentation = tf.keras.Sequential([
            tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
            tf.keras.layers.experimental.preprocessing.RandomRotation(0.1),
        ])
        # Normalization layer to scale inputs from [0, 255] to [-1, +1]
        self.norm_layer = tf.keras.layers.experimental.preprocessing.Normalization()
        mean = np.array([127.5, 127.5, 127.5])
        var = mean ** 2
        self.norm_layer.set_weights([mean, var])

        # Load pre-trained Xception base model without top classifier layers
        # This layer is frozen initially (not trainable)
        self.base_model = tf.keras.applications.Xception(
            weights="imagenet",
            input_shape=(150, 150, 3),
            include_top=False,
        )
        self.base_model.trainable = False  # freeze base model weights

        # Head layers on top of base model
        self.global_avg_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.dropout = tf.keras.layers.Dropout(0.2)
        self.classifier = tf.keras.layers.Dense(1)
    
    def call(self, inputs, training=False):
        # inputs shape: (batch_size, 150, 150, 3)
        x = self.data_augmentation(inputs, training=training)
        x = self.norm_layer(x)
        # base_model runs in inference mode to preserve batch norm statistics
        x = self.base_model(x, training=False)  
        x = self.global_avg_pool(x)
        x = self.dropout(x, training=training)
        outputs = self.classifier(x)
        return outputs

def my_model_function():
    """
    Return an instance of MyModel.
    """
    return MyModel()

def GetInput():
    """
    Return a random tensor input that matches the input expected by MyModel.
    The input shape is (batch_size, 150, 150, 3) with dtype float32.
    Values simulate images with pixel range [0,255].
    """
    batch_size = 32  # typical training batch size in example
    input_shape = (batch_size, 150, 150, 3)
    # Generate random float32 tensor with range 0-255 to simulate raw image pixels
    return tf.random.uniform(input_shape, minval=0, maxval=255, dtype=tf.float32)

