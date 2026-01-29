# tf.random.uniform((B, 28, 28), dtype=tf.float32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # This model corresponds to the MNIST LSTM model in the issue,
        # without CuDNNLSTM usage to avoid the "No OpKernel" error on CPU-only.
        # Input shape: (28, 28) - flattened pen-stroke like sequence per image row.
        self.lstm1 = tf.keras.layers.LSTM(128, activation='relu', return_sequences=True)
        self.dropout1 = tf.keras.layers.Dropout(0.2)
        self.lstm2 = tf.keras.layers.LSTM(128, activation='relu')
        self.dropout2 = tf.keras.layers.Dropout(0.1)
        self.dense1 = tf.keras.layers.Dense(32, activation='relu')
        self.dropout3 = tf.keras.layers.Dropout(0.2)
        self.dense2 = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, inputs, training=False):
        # Forward pass replicating the original Sequential model
        x = self.lstm1(inputs)
        x = self.dropout1(x, training=training)
        x = self.lstm2(x)
        x = self.dropout2(x, training=training)
        x = self.dense1(x)
        x = self.dropout3(x, training=training)
        return self.dense2(x)


def my_model_function():
    # Return an instance of MyModel; weights uninitialized (random init)
    return MyModel()


def GetInput():
    # Generate random batch of data shaped like MNIST sequences for LSTM
    # MNIST dataset shape after load: (60000, 28, 28)
    # Here B=1 (batch size 1), H=28 time steps, W=28 features per time step
    # dtype float32 matching the original normalized input
    return tf.random.uniform((1, 28, 28), dtype=tf.float32)

