# tf.random.uniform((B, 28, 28, 1), dtype=tf.float32) ← Input shape: MNIST grayscale images with shape (28,28,1)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, num_classes=10):
        super().__init__()
        # Define the layers as in the example MNIST CNN model
        self.conv1 = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')
        self.pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.dropout1 = tf.keras.layers.Dropout(0.25)
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dropout2 = tf.keras.layers.Dropout(0.5)
        self.out = tf.keras.layers.Dense(num_classes, activation='softmax')
    
    def call(self, inputs, training=False):
        # Forward pass replicating the architecture in code chunks
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.dropout1(x, training=training)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout2(x, training=training)
        output = self.out(x)
        return output

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a batch of random MNIST-like grayscale images
    # Batch size here arbitrarily chosen as 64 (typical batch size used in examples)
    batch_size = 64
    return tf.random.uniform((batch_size, 28, 28, 1), dtype=tf.float32)

