# tf.random.uniform((1, 50, 50, 1), dtype=tf.float32) ← Inferred input shape based on Conv2D input_shape and image processing

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Layer definitions based on the provided Sequential model
        self.conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(50, 50, 1))
        self.pool1 = tf.keras.layers.MaxPooling2D((2, 2))
        self.conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')
        self.pool2 = tf.keras.layers.MaxPooling2D((2, 2))
        self.conv3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(1, activation='softmax')  # As in original code, but note softmax with 1 unit is uncommon

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    # Shape: (batch_size=1, height=50, width=50, channels=1)
    # Use floats and values between 0 and 1 to resemble normalized grayscale image
    return tf.random.uniform((1, 50, 50, 1), dtype=tf.float32)

