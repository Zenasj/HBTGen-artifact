# tf.random.uniform((B, 224, 224, 3), dtype=tf.float32)
import tensorflow as tf
import numpy as np

class MyModel(tf.keras.Model):
    def __init__(self, num_classes=8):
        super(MyModel, self).__init__()
        # Load pre-trained VGG16 base (include_top=False), freeze weights
        base_model = tf.keras.applications.VGG16(include_top=False,
                                                 weights='imagenet',
                                                 input_shape=(224, 224, 3),
                                                 pooling='avg')
        base_model.trainable = False  # freeze all layers

        self.base_model = base_model
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(512, activation='relu')
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.output_layer = tf.keras.layers.Dense(num_classes, activation='sigmoid')

    def call(self, inputs, training=False):
        # inputs shape: (batch_size, 224, 224, 3)
        x = self.base_model(inputs, training=False)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.batch_norm(x, training=training)
        x = self.dropout(x, training=training)
        x = self.output_layer(x)
        return x

def my_model_function():
    # Return an instance of MyModel with default num_classes=8
    return MyModel(num_classes=8)

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    # Based on the dataset, input images are 224x224x3 RGB images with float32 dtype
    # Values roughly scaled between 0 and 255 in the original data, converted to float32.

    batch_size = 10  # typical batch size used in the original example
    input_shape = (batch_size, 224, 224, 3)
    # Generate random float32 inputs in [0,255), as in original dummy data
    input_tensor = tf.random.uniform(input_shape, minval=0, maxval=255, dtype=tf.float32)
    return input_tensor

