# tf.random.uniform((B, 28, 28), dtype=tf.float32) ← Input shape inferred from Fashion MNIST grayscale images

import tensorflow as tf
from tensorflow import keras

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Model architecture based on issue's Fashion MNIST 3-layer neural network:
        # Flatten input 28x28 -> 784
        self.flatten = keras.layers.Flatten(input_shape=(28, 28), name="Input")
        # Hidden layer with 128 units, ReLU activation
        self.dense1 = keras.layers.Dense(128, activation="relu", name="DataParser")
        # Output layer with 10 units (10 classes), softmax activation
        self.output_layer = keras.layers.Dense(10, activation="softmax", name="Output")

    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.dense1(x)
        return self.output_layer(x)

def my_model_function():
    # Return an instance of MyModel
    model = MyModel()
    # Compile the model with Adam optimizer and sparse categorical crossentropy loss, same as in the issue
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def GetInput():
    # Return a batch of random normalized inputs matching Fashion MNIST input shape
    # Normalize to [0, 1] as in the code snippet in the issue
    batch_size = 1  # batch size 1 used in original training
    return tf.random.uniform((batch_size, 28, 28), minval=0, maxval=1, dtype=tf.float32)

