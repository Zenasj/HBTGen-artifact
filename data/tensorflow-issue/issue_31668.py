# tf.random.uniform((B, 128, 128, 1), dtype=tf.float32)  ← Input shape inferred from the original model input shape (128,128,1)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define the layers as per the provided model
        self.conv = tf.keras.layers.Conv2D(4, (3, 3))
        self.bn = tf.keras.layers.BatchNormalization()
        self.act = tf.keras.layers.Activation('relu')
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(5, activation='softmax')

    def call(self, inputs, training=False):
        # Forward pass, training param used for batchnorm behavior
        x = self.conv(inputs)
        x = self.bn(x, training=training)  # ensure batch norm behaves correctly in training/inference
        x = self.act(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Generate a random input tensor matching (batch_size, 128, 128, 1)
    # Assume batch size 1 for simplicity
    return tf.random.uniform((1, 128, 128, 1), dtype=tf.float32)

