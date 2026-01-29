# tf.random.uniform((8, 2, 5), dtype=tf.float32) ← Input shape inferred from example: batch=8, time=2, features=5
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # LSTM with 10 units, as per example in the issue
        self.lstm = tf.keras.layers.LSTM(10)

    def call(self, inputs, initial_state=None):
        # Pass initial_state explicitly to the LSTM layer if provided
        # This addresses the feature request: passing call arguments to individual layers
        return self.lstm(inputs, initial_state=initial_state)

def my_model_function():
    # Return an instance of MyModel with default initialization
    return MyModel()

def GetInput():
    # Provide a random input tensor matching the example input shape: (batch=8, time=2, features=5)
    # Return a tuple (inputs, initial_state) to match the expected call signature of MyModel
    batch = 8
    time_steps = 2
    features = 5
    lstm_units = 10

    # Input tensor: random float32 values
    inputs = tf.random.uniform((batch, time_steps, features), dtype=tf.float32)

    # Initial states: two tensors of shape (batch, lstm_units) for LSTM cell and hidden states
    h0 = tf.zeros((batch, lstm_units), dtype=tf.float32)
    c0 = tf.zeros((batch, lstm_units), dtype=tf.float32)
    initial_state = [h0, c0]

    return inputs, initial_state

