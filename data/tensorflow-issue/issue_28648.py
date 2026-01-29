# tf.random.uniform((1, 1), dtype=tf.float32) ← Input shape inferred from model input_shape=(1,)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Simple single Dense layer as per the example: Dense(1, input_shape=(1,))
        self.dense = tf.keras.layers.Dense(1)

    def call(self, inputs, training=False):
        return self.dense(inputs)

def my_model_function():
    # Instantiate the model and compile it similarly to the original example
    model = MyModel()
    model.compile(loss="mae", optimizer="adam")
    return model

def GetInput():
    # Return a random input tensor matching the expected model input shape (batch_size=1, features=1)
    return tf.random.uniform((1, 1), dtype=tf.float32)

