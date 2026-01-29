# tf.random.uniform((B, 10), dtype=tf.float32)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Simple 2-layer dense model similar to the example in the issue
        self.dense1 = tf.keras.layers.Dense(2, activation="relu")
        self.dense2 = tf.keras.layers.Dense(1)

    def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)

def my_model_function():
    # Instantiate the model and compile with loss and metrics as in the issue
    model = MyModel()
    # Compile with mse loss and metrics mae and mse, reproducing the behavior discussed
    model.compile(loss="mse", optimizer="sgd", metrics=["mae", "mse"])
    return model

def GetInput():
    # Generate a random input tensor matching the model's expected input shape (batch size 32)
    # Input shape: (batch_size, input_features) == (32, 10)
    return tf.random.uniform((32, 10), dtype=tf.float32)

