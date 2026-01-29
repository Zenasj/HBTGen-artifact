# tf.random.uniform((B, 28, 28, 1), dtype=tf.float32) ← Input shape inferred from issue's Fashion MNIST example

import tensorflow as tf
from tensorflow.keras import layers

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Based on the provided model architecture from the issue description:
        # Input shape (28,28,1), Conv2D layers with ReLU, Dropout, Flatten, Dense with swish (noted swish caused issue,
        # but we keep original activation as in snippet), then output Dense without activation (logits).
        self.conv1 = layers.Conv2D(32, (3, 3), activation='relu')
        self.dropout1 = layers.Dropout(0.4)
        self.conv2 = layers.Conv2D(64, (3, 3), activation='relu')
        self.dropout2 = layers.Dropout(0.4)
        self.conv3 = layers.Conv2D(128, (3, 3), activation='relu')
        self.dropout3 = layers.Dropout(0.4)
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(128, activation='swish')  # original layer activation
        self.dropout4 = layers.Dropout(0.5)
        self.dense_out = layers.Dense(10)  # logits, no activation
        
    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.dropout1(x, training=training)
        x = self.conv2(x)
        x = self.dropout2(x, training=training)
        x = self.conv3(x)
        x = self.dropout3(x, training=training)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout4(x, training=training)
        x = self.dense_out(x)        
        return x

def my_model_function():
    # Instantiate the model
    return MyModel()

def GetInput():
    # Return a random float32 tensor shaped (batch, height, width, channels)
    # The batch size is set arbitrarily to 16 to match the fit batch size from the issue example
    batch_size = 16
    return tf.random.uniform((batch_size, 28, 28, 1), dtype=tf.float32)

