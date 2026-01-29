# tf.random.uniform((B, 2), dtype=tf.float32)  ← Based on example_data=tf.constant([[1.0, 2.0]])

import tensorflow as tf
from tensorflow.keras.layers import Dense

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Define layers as in the reported example
        self.d1 = Dense(2, activation='relu')
        self.d2 = Dense(2, activation='softmax')

    def call(self, x):
        # Mimic the original call function with print replaced by tf.print for graph mode
        # Note: The original example uses @tf.function, which is implicit here.
        tf.print("Tracing the model")
        x = self.d1(x)
        return self.d2(x)


def my_model_function():
    # Return an initialized instance of MyModel
    model = MyModel()
    # Optional: run a dummy input once to build model weights if needed
    dummy_input = tf.random.uniform((1, 2), dtype=tf.float32)
    _ = model(dummy_input)
    return model


def GetInput():
    # Return a random tensor input that matches the expected input of shape (B, 2)
    # Assuming batch size 1 for simplicity here
    return tf.random.uniform((1, 2), dtype=tf.float32)

