# tf.random.uniform((1000, 10), dtype=tf.float32) ← input shape inferred from the example code (X_train shape)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # A simple model: single Dense layer with 1 output unit as in the original minimal example
        self.dense = tf.keras.layers.Dense(1)

    def call(self, inputs, training=False):
        # Forward pass: simple dense layer
        return self.dense(inputs)

def my_model_function():
    # Instantiate and compile the model as in the original TF Keras example
    model = MyModel()
    # Compile with mean squared error loss and SGD optimizer to match original snippet
    model.compile(loss="mse", optimizer="sgd")
    return model

def GetInput():
    # Generate a random tensor input matching the expected shape (batch 1000, features 10)
    # Match dtype to default tf.keras layers usage (float32)
    return tf.random.uniform((1000, 10), dtype=tf.float32)

