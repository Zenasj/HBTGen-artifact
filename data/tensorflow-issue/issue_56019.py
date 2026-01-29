# tf.random.uniform((B, 200, 256, 3), dtype=tf.float32) ← input shape inferred from the issue description and code examples

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Replicating the model architecture from the issue which uses Conv2D layers and Dense layers for classification into 20 classes.
        self.conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(200, 256, 3))
        self.pool1 = tf.keras.layers.MaxPooling2D(2, 2)
        self.conv2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')
        self.pool2 = tf.keras.layers.MaxPooling2D(2, 2)
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(20, activation='softmax')
    
    def call(self, inputs, training=False):
        # Forward pass through the model replicating the original network's flow
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return x

def my_model_function():
    # Returns an instance of MyModel; this model is ready for training/compilation.
    return MyModel()

def GetInput():
    # Creates a random float32 tensor matching the model's expected input:
    # batch size 2 (matching the batch size used in problematic training runs in the issue),
    # height 200, width 256, channels 3 (RGB)
    input_tensor = tf.random.uniform(shape=(2, 200, 256, 3), dtype=tf.float32)
    return input_tensor

