# tf.random.uniform((B=31, H=6, W=1, C=1), dtype=tf.float32) ← inferred input shape based on trainDataForPrediction shape (31, 6, 1)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, hidden_units=9, dense_units=6, activation=['relu', 'sigmoid']):
        super().__init__()
        # LSTM layer with specified units and activation; input shape inferred later from input tensor shape
        self.lstm = tf.keras.layers.LSTM(hidden_units, activation=activation[0])
        # Three Dense layers with specified units and activation
        self.dense1 = tf.keras.layers.Dense(units=dense_units, activation=activation[1])
        self.dense2 = tf.keras.layers.Dense(units=dense_units, activation=activation[1])
        self.dense3 = tf.keras.layers.Dense(units=dense_units, activation=activation[1])

    def call(self, inputs, training=False):
        # Forward pass: LSTM → Dense → Dense → Dense
        x = self.lstm(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        return x

def my_model_function():
    # Instantiate the model with default parameters matching the example code
    return MyModel()

def GetInput():
    # Based on the provided trainDataForPrediction shape of (31, 6, 1)
    # The model expects input shape of (batch_size, timesteps=6, features=1)
    # Generate a random float32 tensor with a batch size of 31 (same as original data)
    batch_size = 31
    timesteps = 6
    features = 1
    return tf.random.uniform((batch_size, timesteps, features), dtype=tf.float32)

