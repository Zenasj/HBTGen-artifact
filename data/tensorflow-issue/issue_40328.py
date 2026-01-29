# tf.random.uniform((1, 10, 10, 10, 5), dtype=tf.float32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Bidirectional ConvLSTM2D layer with filters=16, kernel_size=(1,1), 
        # return_sequences=True and return_state=True as per original code.
        self.lstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.ConvLSTM2D(
                filters=16, 
                kernel_size=(1, 1), 
                return_sequences=True, 
                return_state=True))
        # Dense layers with units 16, 16, and 10 as in original hidden_units list
        self.dense_layers = [
            tf.keras.layers.Dense(16),
            tf.keras.layers.Dense(16),
            tf.keras.layers.Dense(10)
        ]

    def call(self, inputs, training=None, mask=None):
        # Forward pass
        # The Bidirectional ConvLSTM2D returns output and 4 states (h,c) for each direction.
        # We'll only use the sequence output (x).
        # The original code unpacks as: x, _, _, _, _ = self.lstm(x)
        x, *_ = self.lstm(inputs)
        for dense in self.dense_layers:
            x = dense(x)
        return x

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random tensor input matching the expected input shape of (batch, timesteps, height, width, channels)
    # From original code: (1, 10, 10, 10, 5)
    return tf.random.uniform((1, 10, 10, 10, 5), dtype=tf.float32)

