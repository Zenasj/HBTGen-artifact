# tf.random.uniform((B, 128, 1), dtype=tf.float32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, units=128):
        super().__init__()
        # Following the description:
        # Two LSTM layers with dropout and a final dense layer producing output of shape (10,)
        self.lstm1 = tf.keras.layers.LSTM(units, return_sequences=True)
        self.dropout1 = tf.keras.layers.Dropout(0.2)
        self.lstm2 = tf.keras.layers.LSTM(units)
        self.dropout2 = tf.keras.layers.Dropout(0.2)
        self.dense = tf.keras.layers.Dense(10)
    
    def call(self, inputs, training=False):
        x = self.lstm1(inputs)
        x = self.dropout1(x, training=training)
        x = self.lstm2(x)
        x = self.dropout2(x, training=training)
        output = self.dense(x)
        return output

def my_model_function():
    # Default units chosen based on the code snippet and default from tuner range midpoint
    return MyModel(units=128)

def GetInput():
    # Return a random float32 tensor matching the expected input shape (batch_size, 128, 1)
    
    # Batch size is arbitrary; using batch size 32 as common default
    batch_size = 32
    # Input shape required: (128 timesteps, 1 feature)
    input_shape = (batch_size, 128, 1)
    
    # Use tf.random.uniform to generate float32 data in range [0, 1)
    return tf.random.uniform(input_shape, dtype=tf.float32)

