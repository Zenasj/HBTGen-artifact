# tf.random.uniform((B, 8, 900), dtype=tf.float32) ← input shape inferred from numpy random input X: (10000, 8, 900)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Model layers based on get_compiled_cnn_model from issue

        # Input shape: (8, 900) without channel
        # Reshape to (8, 900, 1)
        # Conv2D layer: 10 filters, kernel (2,10), padding same, ReLU activation
        self.reshape = layers.Reshape((8, 900, 1))
        self.batchnorm1 = layers.BatchNormalization()
        self.conv2d = layers.Conv2D(10, (2, 10), padding="same", activation='relu')
        self.flatten = layers.Flatten()
        self.batchnorm2 = layers.BatchNormalization()
        self.dropout1 = layers.Dropout(0.25)
        self.dense1 = layers.Dense(10)
        self.dropout2 = layers.Dropout(0.25)
        self.output_layer = layers.Dense(2, activation='softmax')

    def call(self, inputs, training=False):
        x = self.reshape(inputs)
        x = self.batchnorm1(x, training=training)
        x = self.conv2d(x)
        x = self.flatten(x)
        x = self.batchnorm2(x, training=training)
        x = self.dropout1(x, training=training)
        x = self.dense1(x)
        x = self.dropout2(x, training=training)
        return self.output_layer(x)


def my_model_function():
    model = MyModel()
    # Compile model according to the original function
    loss = keras.losses.CategoricalCrossentropy()
    optimizer = keras.optimizers.Adam(1e-3)
    model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])
    return model


def GetInput():
    # Return a random tensor input matching shape (batch_size, 8, 900)
    # Use batch size 32 as a typical example
    return tf.random.uniform((32, 8, 900), dtype=tf.float32)

