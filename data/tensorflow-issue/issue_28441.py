# tf.random.uniform((B, 28, 28, 1), dtype=tf.float32) ← Input shape matches MNIST grayscale images

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define layers as per the example MNIST ConvNet from the issue
        self.conv1 = tf.keras.layers.Conv2D(16, 8, strides=2, padding='same', activation='relu')
        self.pool1 = tf.keras.layers.MaxPool2D(2, 1)
        self.conv2 = tf.keras.layers.Conv2D(32, 4, strides=2, padding='valid', activation='relu')
        self.pool2 = tf.keras.layers.MaxPool2D(2, 1)
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(32, activation='relu')
        self.dense2 = tf.keras.layers.Dense(10, activation='softmax')  # Output layer with softmax
    
    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        return self.dense2(x)

def my_model_function():
    # Return an instance of MyModel
    # Note: weights are not pre-loaded, user may train after instantiation
    return MyModel()

def GetInput():
    # Return a random input tensor matching the expected input shape (batch 1000 as in the example)
    # Using float32 to match typical image input dtype
    BATCH_SIZE = 1000  # Mimicking batch size used in example training
    input_tensor = tf.random.uniform((BATCH_SIZE, 28, 28, 1), dtype=tf.float32)
    return input_tensor

