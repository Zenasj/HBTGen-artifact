# tf.random.uniform((None, 10), dtype=tf.float32) ← Inferred input shape: batch size unknown, 10 features (assumed)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # A tiny toy model with two Dense layers as per the original minimal example
        self.dense1 = tf.keras.layers.Dense(4, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(5, activation=tf.nn.softmax)

    def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)

def my_model_function():
    # Initialize and return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random float32 tensor with a batch dimension compatible with MyModel's input
    # The original code does not specify input dimension explicitly; typical Dense input needs shape (..., features)
    # From dense1 input shape: we can infer input features = 10 (arbitrary small number for toy example)
    # We use batch size = 1 here to keep it simple and efficient.
    batch_size = 1
    input_features = 10
    return tf.random.uniform(shape=(batch_size, input_features), dtype=tf.float32)

