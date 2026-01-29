# tf.random.uniform((B, 99, 1), dtype=tf.float32) ← Input shape inferred from sequence_length=100, input to model is (sequence_length - 1, 1)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # LSTM layers as described, matching the sequential model structure
        self.lstm1 = tf.keras.layers.LSTM(
            units=32, return_sequences=True, input_shape=(99, 1))
        self.dropout1 = tf.keras.layers.Dropout(0.2)
        self.lstm2 = tf.keras.layers.LSTM(
            units=128, return_sequences=True)
        self.dropout2 = tf.keras.layers.Dropout(0.2)
        self.lstm3 = tf.keras.layers.LSTM(
            units=100, return_sequences=False)
        self.dropout3 = tf.keras.layers.Dropout(0.2)
        self.dense = tf.keras.layers.Dense(units=1)
        self.activation = tf.keras.layers.Activation('linear')

    def call(self, inputs, training=False):
        x = self.lstm1(inputs)
        x = self.dropout1(x, training=training)
        x = self.lstm2(x)
        x = self.dropout2(x, training=training)
        x = self.lstm3(x)
        x = self.dropout3(x, training=training)
        x = self.dense(x)
        x = self.activation(x)
        return x

def my_model_function():
    # Return a compiled instance of MyModel
    model = MyModel()
    # Compile with same loss and optimizer as in original Sequential model
    model.compile(loss='mean_squared_error', optimizer='rmsprop')
    return model

def GetInput():
    # Generates a random input tensor shaped (batch_size, sequence_length-1, 1)
    # Use batch_size = 50 as in original code
    batch_size = 50
    sequence_length = 100
    input_shape = (batch_size, sequence_length - 1, 1)
    return tf.random.uniform(input_shape, dtype=tf.float32)

