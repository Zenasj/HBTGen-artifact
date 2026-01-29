# tf.random.uniform((B, 32, 32, 3), dtype=tf.float32) ← Input shape inferred from Conv2D input_shape

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Replicating the Sequential model structure as layers in a subclassed Model
        self.conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3))
        self.pool1 = tf.keras.layers.MaxPooling2D((2, 2))
        self.conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')
        self.pool2 = tf.keras.layers.MaxPooling2D((2, 2))
        self.conv3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(10)  # output logits for 10 classes

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.dense1(x)
        logits = self.dense2(x)
        return logits


def my_model_function():
    # Instantiate and compile the model similarly to original sequential model
    model = MyModel()
    # Compile here to reflect original loss and optimizer setup
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    return model


def GetInput():
    # Return a random input tensor matching (B, 32, 32, 3)
    # Using batch size = 8 as a reasonable example
    batch_size = 8
    # Input dtype inferred from typical conv2D use: float32
    return tf.random.uniform((batch_size, 32, 32, 3), dtype=tf.float32)

