# tf.random.uniform((B, 1), dtype=tf.float32) ← Input shape inferred as [batch_size, 1] based on example

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Mimicking the sequential model from the issue: Dense(1, input_shape=(1,)), Dense(10), Dense(1)
        self.dense1 = tf.keras.layers.Dense(1)
        self.dense2 = tf.keras.layers.Dense(10)
        self.dense3 = tf.keras.layers.Dense(1)

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        return x

def my_model_function():
    # Create model and compile to replicate behavior close to the issue example
    model = MyModel()
    # Build the model by calling it once with dummy input to initialize weights
    _ = model(tf.zeros((1,1)))
    model.compile(optimizer='sgd', loss='mse')
    return model

def GetInput():
    # Return a random tensor matching input shape [batch_size, 1], batch size = 2 as in example
    return tf.random.uniform((2, 1), dtype=tf.float32)

