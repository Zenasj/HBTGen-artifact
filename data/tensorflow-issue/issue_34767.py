# tf.random.uniform((64, 3, 2), dtype=tf.float32) ← inferred from the example input shape used in the issue

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Recreate the Sequential LSTM model from the issue scenario 2 which had issues with retraining after loading
        # This is a simple LSTM layer with 10 units returning sequences, followed by a Dense layer projecting to 2 features.
        self.lstm = tf.keras.layers.LSTM(10, return_sequences=True)
        self.projection = tf.keras.layers.Dense(2)

    def call(self, inputs, training=False):
        x = self.lstm(inputs, training=training)
        x = self.projection(x)
        return x

def my_model_function():
    # Return an instance of MyModel, compiled with Adam and MSE loss to match the original training setup
    model = MyModel()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mean_squared_error'
    )
    return model

def GetInput():
    # Return a random tensor input that matches the expected input shape: batch=64, timesteps=3, features=2
    return tf.random.uniform((64, 3, 2), dtype=tf.float32)

