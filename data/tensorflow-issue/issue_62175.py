# tf.random.uniform((1, 224, 224, 3), dtype=tf.float32)

import tensorflow as tf
from tensorflow.keras import layers

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define layers roughly matching the original model from the issue
        self.sepconv1 = layers.SeparableConv2D(32, (3,3))
        self.bn1 = layers.BatchNormalization()
        self.relu1 = layers.ReLU()
        self.pool1 = layers.MaxPooling2D(pool_size=(2,2))

        self.sepconv2 = layers.SeparableConv2D(64, (3,3))
        self.bn2 = layers.BatchNormalization()
        self.relu2 = layers.ReLU()
        self.pool2 = layers.MaxPooling2D(pool_size=(2,2))

        # The original code ended with a Dense layer after the pooling over 4D feature map.
        # Dense expects 2D input (batch, features), so flatten or global pooling is needed.
        # The original snippet applied Dense directly to pool2 output which is spatial.
        # We infer that the Dense layer was applied directly on output with spatial dims,
        # likely intended as a 1x1 conv or a flatten first.
        # We'll flatten here to match typical usage.
        self.flatten = layers.Flatten()
        self.dense = layers.Dense(4, activation='softmax')

    def call(self, inputs, training=False):
        x = self.sepconv1(inputs)
        x = self.bn1(x, training=training)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.sepconv2(x)
        x = self.bn2(x, training=training)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.flatten(x)
        out = self.dense(x)
        return out

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random tensor shaped (1, 224, 224, 3) matching the model's input
    # Use a fixed seed for reproducibility
    tf.random.set_seed(123)
    return tf.random.uniform((1, 224, 224, 3), dtype=tf.float32)

