# tf.random.uniform((1, None, 600), dtype=tf.float32)  # Input shape corresponds to (batch_size=1, time_steps variable, features=600)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Reconstructing the original sequential model architecture as described:
        # Masking layer with mask_value = -27/255
        self.masking = tf.keras.layers.Masking(mask_value=-27/255, input_shape=(None, 600))
        # Two stacked LSTM layers, first returns sequences, second does not
        self.lstm1 = tf.keras.layers.LSTM(32, return_sequences=True)
        self.lstm2 = tf.keras.layers.LSTM(32, return_sequences=False)
        # Dropout layer with rate 0.5 after LSTM layers
        self.dropout1 = tf.keras.layers.Dropout(0.5)
        # Dense layer with 16 units and tanh activation
        self.dense1 = tf.keras.layers.Dense(16, activation='tanh')
        # Another dropout layer with rate 0.5
        self.dropout2 = tf.keras.layers.Dropout(0.5)
        # Output Dense layer with 4 units and softmax activation (4-class classification)
        self.dense2 = tf.keras.layers.Dense(4, activation='softmax')

    def call(self, inputs, training=False):
        x = self.masking(inputs)
        x = self.lstm1(x)
        x = self.lstm2(x)
        x = self.dropout1(x, training=training)
        x = self.dense1(x)
        x = self.dropout2(x, training=training)
        output = self.dense2(x)
        return output

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Based on the preprocessing from the issue:
    # input shape is (1, variable_length, 600) with float32 dtype and values roughly normalized by 255 after padding with -27
    # For testing, create a random float tensor with one time step (e.g., 10) for stability
    batch_size = 1
    time_steps = 10  # arbitrary small length for example
    features = 600
    # Generate random float inputs between 0 and 1 matching the input range after normalization
    # Original padding used value -27, normalized by 255 -> approximately -0.10588
    # To keep similar data distribution, generate from around [-0.1, 1.0]
    import numpy as np
    input_data = np.random.uniform(low=-0.1, high=1.0, size=(batch_size, time_steps, features)).astype(np.float32)
    return tf.convert_to_tensor(input_data)

