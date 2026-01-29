# tf.random.uniform((B, 32, 32, 3), dtype=tf.float32) ← CIFAR-100 images are 32x32 RGB images

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define the architecture based on the user's reported model.
        # Input shape (32,32,3)
        self.conv1 = tf.keras.layers.Conv2D(32, 5, padding='same', activation='relu', input_shape=(32, 32, 3))
        self.pool1 = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()

        self.conv2 = tf.keras.layers.Conv2D(64, 5, padding='same', activation='relu')
        self.pool2 = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same')

        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(1024, activation='relu')
        self.dropout = tf.keras.layers.Dropout(0.4)
        self.dense2 = tf.keras.layers.Dense(10, activation='softmax')  # num_classes=10 as per the code snippet

    def call(self, inputs, training=False, mask=None):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.bn1(x, training=training)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        return self.dense2(x)


def my_model_function():
    # Return an instance of MyModel
    return MyModel()


def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    # Batch size is randomly chosen to 4 for demonstration; can be any positive integer.
    batch_size = 4
    # CIFAR-100 input shape: 32x32x3, dtype float32 (pixel values between 0-1).
    return tf.random.uniform((batch_size, 32, 32, 3), dtype=tf.float32)

