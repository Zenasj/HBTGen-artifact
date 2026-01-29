# tf.random.uniform((B, 28, 28, 1), dtype=tf.float32) ← Input shape inferred from MNIST dataset grayscale images

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define the convolutional regression model architecture as described
        self.conv1 = tf.keras.layers.Conv2D(32, 3, activation='relu')
        self.pool1 = tf.keras.layers.MaxPooling2D(3)
        self.conv2 = tf.keras.layers.Conv2D(64, 3, activation='relu')
        self.pool2 = tf.keras.layers.MaxPooling2D(3)
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.output_layer = tf.keras.layers.Dense(1, activation='linear')

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        output = self.output_layer(x)
        return output

def my_model_function():
    # Create an instance of MyModel with compiled configuration for regression
    model = MyModel()
    model.compile(optimizer='adam', loss='mae', metrics=['mae'])
    return model

def GetInput():
    # Generate a random tensor mimicking MNIST image input: batch size=1
    # Values normalized similar to original dataset (0 to 1 float32)
    return tf.random.uniform((1, 28, 28, 1), minval=0.0, maxval=1.0, dtype=tf.float32)

