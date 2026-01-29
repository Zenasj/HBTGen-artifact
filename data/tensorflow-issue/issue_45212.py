# tf.random.uniform((B, 32, 32, 3), dtype=tf.float32) ← CIFAR-10 image shape (batch size B is dynamic)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Build the CNN model similar to TensorFlow CNN Tutorial for CIFAR-10
        
        # Conv2D(32, 3x3), relu activation, input shape (32, 32, 3)
        self.conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')
        self.pool1 = tf.keras.layers.MaxPooling2D((2, 2))
        
        # Conv2D(64, 3x3), relu activation
        self.conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')
        self.pool2 = tf.keras.layers.MaxPooling2D((2, 2))
        
        # Conv2D(64, 3x3), relu activation
        self.conv3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')
        
        # Flatten layer
        self.flatten = tf.keras.layers.Flatten()
        
        # Dense 64 units, relu activation
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        
        # Output Dense layer with 10 units (logits for CIFAR-10 classes)
        self.dense_out = tf.keras.layers.Dense(10)

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.dense1(x)
        logits = self.dense_out(x)
        return logits


def my_model_function():
    # Return an instance of MyModel
    return MyModel()


def GetInput():
    # Return a random tensor matching CIFAR-10 input shape:
    # 32x32 RGB images, batch size 1 for testing.
    # Using float32 in range [0, 1] like original input preprocessing.
    return tf.random.uniform((1, 32, 32, 3), minval=0, maxval=1, dtype=tf.float32)

