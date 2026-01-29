# tf.random.uniform((B, 32, 32, 3), dtype=tf.float32) ← CIFAR-10 image input shape

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Construct layers as per the original Sequential model in the issue
        self.conv1 = tf.keras.layers.Conv2D(8, (3, 3), activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(8, (3, 3), activation='relu')
        self.pool1 = tf.keras.layers.MaxPooling2D((2, 2))
        self.drop1 = tf.keras.layers.Dropout(0.25)
        self.conv3 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu')
        self.conv4 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu')
        self.pool2 = tf.keras.layers.MaxPooling2D((2, 2))
        self.drop2 = tf.keras.layers.Dropout(0.25)
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(32, activation='relu')
        self.drop3 = tf.keras.layers.Dropout(0.5)
        self.out = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.drop1(x, training=training)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.drop2(x, training=training)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.drop3(x, training=training)
        return self.out(x)

def my_model_function():
    # Returns an instance of MyModel with default weights
    return MyModel()

def GetInput():
    # CIFAR-10 images are 32x32 RGB images with float32 normalized values
    # Batch size is set to 4 as a reasonable small batch size for quick testing
    batch_size = 4
    height, width, channels = 32, 32, 3
    # Generate random tensor with values between 0 and 1 (normalized input)
    return tf.random.uniform(
        (batch_size, height, width, channels),
        minval=0, maxval=1, dtype=tf.float32)

