# tf.random.uniform((B, 28, 28, 1), dtype=tf.float32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Build the equivalent of the MNIST model from the issue with two Conv2D layers,
        # pooling, dropouts, flatten and dense layers.
        # Input shape assumed: (batch_size, 28, 28, 1) grayscale images from MNIST
        self.conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')
        self.pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.dropout1 = tf.keras.layers.Dropout(0.25)
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dropout2 = tf.keras.layers.Dropout(0.5)
        self.dense2 = tf.keras.layers.Dense(10, activation='softmax', dtype='float32')
        # Note: output dtype forced to float32 for numeric stability with mixed precision

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.dropout1(x, training=training)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout2(x, training=training)
        return self.dense2(x)

def my_model_function():
    # Return a fresh model instance, weights randomly initialized
    return MyModel()

def GetInput():
    # Return a random tensor shaped (batch, height, width, channels) = (128, 28, 28, 1),
    # dtype float32, normalized [0,1]. This shape matches the example MNIST dataset input.
    batch_size = 128  # batch size is 128 as in the example code
    height = 28
    width = 28
    channels = 1
    return tf.random.uniform((batch_size, height, width, channels), dtype=tf.float32)

