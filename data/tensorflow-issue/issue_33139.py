# tf.random.uniform((B, T, 205), dtype=tf.float32) ← Input shape inferred from (num_of_instances, time_length, num_features)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define the layers to match the original Sequential LSTM model
        self.lstm1 = tf.keras.layers.LSTM(75, return_sequences=True)
        self.dropout1 = tf.keras.layers.Dropout(0.3)
        self.lstm2 = tf.keras.layers.LSTM(75, return_sequences=True)
        self.dropout2 = tf.keras.layers.Dropout(0.3)
        self.time_dist_dense = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1))

    def call(self, inputs, training=False):
        # Forward pass through LSTM layers with dropout
        x = self.lstm1(inputs)
        x = self.dropout1(x, training=training)
        x = self.lstm2(x)
        x = self.dropout2(x, training=training)
        x = self.time_dist_dense(x)
        return x


def my_model_function():
    # Instantiate and return the model
    model = MyModel()

    # Compile the model similarly as original
    adam = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='mse', optimizer=adam)

    return model


def GetInput():
    # Generate a random input tensor matching the expected input shape:
    # (batch_size, time_length, num_features) = (5, 12, 205)
    # Batch size 5 chosen arbitrarily for example usage.
    batch_size = 5
    time_length = 12
    num_features = 205
    return tf.random.uniform((batch_size, time_length, num_features), dtype=tf.float32)

