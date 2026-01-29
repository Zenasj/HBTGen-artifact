# tf.random.uniform((B, 1), dtype=tf.float32)  ← Input shape: batch size B, 1 feature scalar inputs

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Single Dense layer mapping input of shape (batch, 1) to output (batch, 1)
        self.dense = tf.keras.layers.Dense(units=1)

    def call(self, inputs, training=False):
        return self.dense(inputs)

def my_model_function():
    # Return an instance of MyModel
    model = MyModel()
    # Per the original code in the issue, the model expects to be compiled with sgd optimizer and mse loss
    model.compile(optimizer='sgd', loss='mean_squared_error')
    return model

def GetInput():
    # Create input tensor matching the input shape for MyModel: (batch_size, 1)
    # Using batch size of 6 to match the example
    x = tf.constant([[-1.0], [0.0], [1.0], [2.0], [3.0], [4.0]], dtype=tf.float32)
    return x

