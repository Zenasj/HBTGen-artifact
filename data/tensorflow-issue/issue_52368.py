# tf.random.uniform((B, 32), dtype=tf.float32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Single Dense layer as per the reproduced minimal example
        self.dense = tf.keras.layers.Dense(1)

    def call(self, inputs):
        return self.dense(inputs)

def my_model_function():
    # Construct the model and compile with Adam optimizer and MSE loss,
    # reflecting the example in the issue.
    model = MyModel()
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def GetInput():
    # Generate a random input tensor with batch size 128 and feature size 32,
    # matching the example used to train the model.
    return tf.random.uniform(shape=(128, 32), dtype=tf.float32)

