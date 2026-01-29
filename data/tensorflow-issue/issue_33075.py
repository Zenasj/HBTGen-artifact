# tf.random.uniform((32, 4), dtype=tf.float32) ← Inferred input shape (batch size 32, features 4)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define a simple binary classifier with one Dense layer and sigmoid activation,
        # matching the example in the issue.
        # This is equivalent to:
        # ipt = Input(shape=(4,))
        # out = Dense(1, activation='sigmoid')(ipt)
        # model = Model(ipt, out)
        self.dense = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs, training=False):
        # Forward pass through the dense layer
        return self.dense(inputs)


def my_model_function():
    # Return an instance of MyModel
    return MyModel()


def GetInput():
    # Generate a random input tensor matching shape (32,4) and dtype float32, as per example usage
    # This simulates a batch size of 32 samples, each with 4 features
    return tf.random.uniform(shape=(32, 4), dtype=tf.float32)

