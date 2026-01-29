# tf.random.uniform((B, 28, 28, 1), dtype=tf.float32) ← input shape from MNIST example

import tensorflow as tf
from tensorflow.keras import layers

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Replicating the simple CNN MNIST example from the issue description
        self.conv1 = layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1))
        self.pool1 = layers.MaxPooling2D((2, 2))
        self.conv2 = layers.Conv2D(64, (3, 3), activation='relu')
        self.pool2 = layers.MaxPooling2D((2, 2))
        self.conv3 = layers.Conv2D(64, (3, 3), activation='relu')
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(64, activation='relu')
        self.dense2 = layers.Dense(10, activation='softmax')  # 10 classes for MNIST

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.dense1(x)
        return self.dense2(x)

def customLoss(y_true, y_pred):
    """
    Custom loss function that wraps tf.keras.losses.sparse_categorical_crossentropy exactly.
    This is the minimal working custom loss shown in the GitHub issue example.
    It takes y_true (integer labels) and y_pred (probabilities after softmax)
    and returns the per-sample sparse categorical crossentropy loss.
    """
    return tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)

def my_model_function():
    # Return an instance of MyModel.
    # Note: For training, one would compile the model with the customLoss as the loss function.
    return MyModel()

def GetInput():
    # Return a batch of random images of shape (batch_size, 28, 28, 1) float32 as expected by the model.
    # Labels should be integer class indices (0-9) for sparse categorical crossentropy.
    batch_size = 32
    images = tf.random.uniform((batch_size, 28, 28, 1), minval=0.0, maxval=1.0, dtype=tf.float32)
    labels = tf.random.uniform((batch_size,), minval=0, maxval=10, dtype=tf.int32)
    return images, labels

