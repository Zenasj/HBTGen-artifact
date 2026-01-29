# tf.random.uniform((1, 1), dtype=tf.float32)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Single Dense layer with 1 output unit and linear activation (as in the original example)
        self.dense = tf.keras.layers.Dense(1, activation="linear")

    def call(self, inputs):
        # Forward pass through the dense layer
        return self.dense(inputs)

def my_model_function():
    # Return an instance of MyModel, compiled the same way as original example
    model = MyModel()
    # As in the original code, "sgd" optimizer and "mse" loss were used on compiling
    model.compile(optimizer="sgd", loss="mse")
    return model

def GetInput():
    # Original example used shape (1,1) random uniform input
    # Using tf.float32 dtype (TensorFlow default)
    return tf.random.uniform((1, 1), dtype=tf.float32)

