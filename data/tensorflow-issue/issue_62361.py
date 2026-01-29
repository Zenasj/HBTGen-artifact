# tf.random.uniform((B, 28, 28, 1), dtype=tf.float32) ← Input shape inferred as MNIST grayscale images 28x28x1

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define a CNN model akin to the "model2" referenced in the comments:
        # 3 Conv2D layers with ReLU activations, interleaved with max-pooling,
        # then flatten, dense relu, dropout, and final output dense layer.
        self.conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')
        self.pool1 = tf.keras.layers.MaxPooling2D((2,2))
        self.conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')
        self.pool2 = tf.keras.layers.MaxPooling2D((2,2))
        self.conv3 = tf.keras.layers.Conv2D(64, (3,3), activation='relu')
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dropout = tf.keras.layers.Dropout(0.2)
        self.dense2 = tf.keras.layers.Dense(10, name="y")

    def call(self, x, training=False):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        x = self.dense2(x)
        return x

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random tensor input simulating a batch of images with shape [B, 28, 28, 1]
    # Use batch size 32 as a reasonable example batch
    batch_size = 32
    return tf.random.uniform((batch_size, 28, 28, 1), dtype=tf.float32)

