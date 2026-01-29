# tf.random.uniform((B, 256, 1), dtype=tf.float32) ← Input shape inferred from the LSTM model input

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # LSTM layer matching the reported model: 256 units, no bias, return_sequences=False, input shape (256,1)
        self.lstm = tf.keras.layers.LSTM(256, use_bias=False, return_sequences=False, name='LSTM')
        # Following Dense layers per original snippet
        self.dense1 = tf.keras.layers.Dense(100, activation='relu')
        self.dense2 = tf.keras.layers.Dense(1, activation='linear', name='output')

    def call(self, inputs):
        x = self.lstm(inputs)
        x = self.dense1(x)
        output = self.dense2(x)
        return output

def my_model_function():
    # Return an instance of MyModel
    # Note: weights are initialized randomly, matching the original Keras model before training.
    return MyModel()

def GetInput():
    # Return a random input tensor matching the expected shape (batch_size, 256, 1)
    # Assumptions based on original training: batch size can be flexible (here 32)
    batch_size = 32
    # Generate random float input tensor with proper shape and dtype float32
    return tf.random.uniform((batch_size, 256, 1), dtype=tf.float32)

