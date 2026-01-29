# tf.random.uniform((B, 28, 28), dtype=tf.float32) ← The input shape is (batch_size, 28, 28) grayscale images from MNIST

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Recreate the CNN model architecture described in the issue
        self.reshape = tf.keras.layers.Reshape(target_shape=(28, 28, 1))
        self.conv1 = tf.keras.layers.Conv2D(256, 2, activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(128, 2, activation='relu')
        self.conv3 = tf.keras.layers.Conv2D(32, 1, activation='relu')
        self.conv4 = tf.keras.layers.Conv2D(32, 2, activation='relu')
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(2048, activation='relu')
        self.dense2 = tf.keras.layers.Dense(1024, activation='relu')
        self.dense3 = tf.keras.layers.Dense(128, activation='relu')
        self.out = tf.keras.layers.Dense(10)  # logits for 10 classes

    def call(self, inputs):
        x = self.reshape(inputs)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        logits = self.out(x)
        return logits


def my_model_function():
    # Return an uncompiled instance of MyModel
    # Compilation with loss and optimizer would normally happen in training script / strategy scope
    return MyModel()


def GetInput():
    # Create a random tensor that mimics MNIST input batches with shape (batch_size, 28, 28),
    # dtype float32 normalized [0,1]
    # Use a reasonable batch size for demonstration, e.g., 64
    batch_size = 64
    input_tensor = tf.random.uniform((batch_size, 28, 28), minval=0, maxval=1, dtype=tf.float32)
    return input_tensor

