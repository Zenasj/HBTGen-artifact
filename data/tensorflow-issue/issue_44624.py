# tf.random.uniform((B, None, 28), dtype=tf.float32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # An LSTM layer with 64 units, input shape (None, 28)
        # This corresponds to the setup in the issue, expecting variable sequence length and feature size 28.
        self.lstm = tf.keras.layers.LSTM(64)
        self.dense = tf.keras.layers.Dense(10)

    def call(self, inputs):
        x = self.lstm(inputs)
        x = self.dense(x)
        return x

def my_model_function():
    # Return an instance of MyModel.
    # No pretrained weights provided in the issue, so this is initialized fresh.
    return MyModel()

def GetInput():
    # Returns a random tensor input that matches the input expected by MyModel:
    # Shape: [batch_size, time_steps, features] = [8, 512, 28]
    # The time_steps here is inferred from the benchmark input shape used in the issue's timing tests.
    # dtype float32 is standard for the model.
    return tf.random.uniform((8, 512, 28), dtype=tf.float32)

