# tf.random.uniform((B, 128, 64, 3), dtype=tf.float32) ← Input shape inferred from model Input layer (128,64,3) with batch B

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # The model converts input RGB images to HSV, then applies Conv2D, MaxPooling, Dense, Dropout, and Dense layers.
        # This mirrors the provided keras functional model.
        self.hsv_conversion = tf.keras.layers.Lambda(lambda x: tf.image.rgb_to_hsv(x))
        self.conv2d = tf.keras.layers.Conv2D(16, (5,5), activation='relu', padding='same')
        self.maxpool = tf.keras.layers.MaxPooling2D((2,2), padding='same')
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(32, activation='relu')
        self.dropout = tf.keras.layers.Dropout(0.4)
        self.output_layer = tf.keras.layers.Dense(2, activation='softmax')

    def call(self, inputs, training=False):
        # Forward pass reflecting the original Keras Functional model
        x = self.hsv_conversion(inputs)
        x = self.conv2d(x)
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        x = self.output_layer(x)
        return x

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random tensor input matching the input expected by MyModel:
    # Shape is (batch_size, height=128, width=64, channels=3), dtype float32 in [0,1] range as typical RGB image tensor inputs
    batch_size = 1  # Using batch size 1 as default; can be adjusted as needed
    return tf.random.uniform((batch_size, 128, 64, 3), minval=0.0, maxval=1.0, dtype=tf.float32)

