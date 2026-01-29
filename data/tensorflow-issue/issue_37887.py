# tf.random.uniform((B, 784), dtype=tf.float32) ← input shape inferred from MNIST flattened dataset

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Defining the layers exactly as per the MNIST example
        self.dense_1 = tf.keras.layers.Dense(64, activation='relu', name='dense_1')
        self.dense_2 = tf.keras.layers.Dense(64, activation='relu', name='dense_2')
        self.predictions = tf.keras.layers.Dense(10, activation='softmax', name='predictions')

    def call(self, inputs):
        x = self.dense_1(inputs)
        x_intermediate = self.dense_2(x)
        out_pred = self.predictions(x_intermediate)
        # The original bug is about multi-output models; here we output two tensors:
        # One intermediate tensor (x_intermediate), and the softmax predictions.
        # This mirrors the faulty model output=[x, outputs] from the issue.
        return [x_intermediate, out_pred]


def my_model_function():
    # Return an instance of the model defined above
    return MyModel()

def GetInput():
    # Returns a random input tensor matching the expected input shape
    # Batch size is set to 32 arbitrarily; this can be any number.
    # Inputs are normalized floats in [0,1], shape (batch, 784)
    return tf.random.uniform((32, 784), minval=0, maxval=1, dtype=tf.float32)

