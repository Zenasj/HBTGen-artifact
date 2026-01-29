# tf.random.uniform((B, 28*28), dtype=tf.float32) ← Input shape inferred from EMNIST images reshaped (28x28 images flattened)

import tensorflow as tf
from tensorflow import keras

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Build a simple multilayer perceptron similar to the reported model in the issue
        self.dense1 = keras.layers.Dense(112, activation='relu')
        self.dense2 = keras.layers.Dense(112, activation='relu')
        # Number of output classes for EMNIST is 26 (letters) or 62 depending on split,
        # but we'll assume 47 classes, which is standard for EMNIST Balanced split.
        self.dense3 = keras.layers.Dense(47, activation='softmax')

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        return x

def my_model_function():
    # Instantiate and compile the model similarly as in the reported code
    model = MyModel()
    compile_options = {
        "loss": "categorical_crossentropy",
        "optimizer": "adam",
        "metrics": ["categorical_accuracy"],
        "weighted_metrics": ["categorical_accuracy"]
    }
    model.compile(**compile_options)
    return model

def GetInput():
    # Create random input tensor matching shape: batch_size x input_dim = 2048 x (28*28)
    # Using batch size 2048 as per fit_options in the issue
    batch_size = 2048
    input_dim = 28 * 28  # Flattened image size for EMNIST
    # Return input tensor of shape (batch_size, input_dim), float32 between 0 and 1
    return tf.random.uniform((batch_size, input_dim), dtype=tf.float32)

