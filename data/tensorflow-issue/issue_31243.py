# tf.random.uniform((30, 16000, 1), dtype=tf.float32) ← Input shape based on the provided example of data shape

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # The original model has:
        # 1 LSTM layer with 15 units, return_sequences=True, input_shape=(16000, 1)
        # followed by 8 LSTM layers with 15 units, return_sequences=True
        # finally a Dense(1) layer
        
        self.lstm_layers = []
        # First LSTM layer with input shape specified
        self.lstm_layers.append(
            tf.keras.layers.LSTM(15, return_sequences=True)
        )
        # Add 8 more LSTM layers with return_sequences=True
        for _ in range(8):
            self.lstm_layers.append(
                tf.keras.layers.LSTM(15, return_sequences=True)
            )
        self.dense = tf.keras.layers.Dense(1)

    @tf.function(jit_compile=True)
    def call(self, inputs, training=False):
        x = inputs
        for lstm in self.lstm_layers:
            x = lstm(x, training=training)
        x = self.dense(x)
        return x

def my_model_function():
    # Returns a fresh instance of MyModel.
    # This matches the described solution in the issue:
    # create the model from a function and then manually load weights if needed.
    return MyModel()

def GetInput():
    # Return a random tensor with shape (30,16000,1) matching the described input.
    # Using uniform distribution as a reasonable default.
    return tf.random.uniform((30, 16000, 1), dtype=tf.float32)

