# tf.random.uniform((B, 32, 32, 3), dtype=tf.float32) ← Input shape inferred from the ImageDataGenerator target_size and color_mode

import tensorflow as tf
from tensorflow.keras import layers

class MyModel(tf.keras.Model):
    def __init__(self, num_classes=200):
        super().__init__()
        # The architecture follows the conv + dense layers described in the example:
        # Conv2D(128, (7,7)) -> Conv2D(64, (3,3)) -> Conv2D(32, (3,3)) -> Flatten -> Dense(400, relu) -> Dense(num_classes, softmax)
        self.conv1 = layers.Conv2D(128, (7, 7))
        self.conv2 = layers.Conv2D(64, (3, 3))
        self.conv3 = layers.Conv2D(32, (3, 3))
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(400, activation='relu')
        self.dense2 = layers.Dense(num_classes, activation='softmax')
    
    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.dense1(x)
        return self.dense2(x)

def my_model_function():
    # Instantiate and return the model with the number of classes used in the example (200).
    model = MyModel(num_classes=200)
    # Compile with Adam optimizer and categorical crossentropy loss, matching the example.
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    return model

def GetInput():
    # Create a random tensor matching the expected input shape (batch size 128 is typical from example)
    # but batch size can be flexible, here we pick 128 as per original batch_size in usage
    batch_size = 128
    height = 32
    width = 32
    channels = 3
    # Random float32 tensor scaled to [0, 1) like ImageDataGenerator rescale=1./255
    return tf.random.uniform((batch_size, height, width, channels), dtype=tf.float32)

