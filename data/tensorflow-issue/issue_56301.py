# tf.random.uniform((1, 21 * 3 * 2 * 21), dtype=tf.float32)  # Input shape: (batch=1, features=21*3*2*21)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # The original model layers based on the Sequential model described:
        # Input shape: (21*3*2*21,) i.e. flat vector
        # Reshape to (21, 3*2*21)
        # Then LSTM layers with Dropout, ending with Dense(27, softmax)
        
        self.reshape = tf.keras.layers.Reshape((21, 3*2*21))  # (21, 126)
        # LSTM(8, return_sequences=True, unroll=True), followed by Dropout(0.4)
        # LSTM(8), Dropout(0.5)
        # Dense(27, activation='softmax')
        # unroll=True is deprecated in TF 2.20; still we keep same semantics
        self.lstm1 = tf.keras.layers.LSTM(8, return_sequences=True, unroll=True)
        self.dropout1 = tf.keras.layers.Dropout(0.4, seed=42)
        self.lstm2 = tf.keras.layers.LSTM(8, unroll=True)
        self.dropout2 = tf.keras.layers.Dropout(0.5, seed=42)
        self.dense = tf.keras.layers.Dense(27, activation='softmax')
        
    def call(self, inputs, training=False):
        x = self.reshape(inputs)
        x = self.lstm1(x, training=training)
        x = self.dropout1(x, training=training)
        x = self.lstm2(x, training=training)
        x = self.dropout2(x, training=training)
        x = self.dense(x)
        return x

def my_model_function():
    # Return an instance of MyModel - weights are randomly initialized here,
    # since original weights/model file paths are not available.
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    # The input shape is (batch_size=1, 21*3*2*21=2646)
    batch_size = 1
    input_dim = 21 * 3 * 2 * 21
    return tf.random.uniform(shape=(batch_size, input_dim), dtype=tf.float32)

