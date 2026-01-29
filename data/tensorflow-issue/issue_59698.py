# tf.random.uniform((1, 50, 5), dtype=tf.float32)  ← Input shape inferred from test_seq: (1, 50, 5)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Encoder part
        self.conv1 = tf.keras.layers.Conv1D(filters=32, kernel_size=15, padding='same',
                                            data_format='channels_last', dilation_rate=1,
                                            activation='linear')
        # LSTM layer with unroll=True as suggested in the issue comments to speed up inference on short sequences
        # Activation relu consistent with original model
        self.lstm1 = tf.keras.layers.LSTM(units=50, activation='relu', return_sequences=False, unroll=True)
        self.dropout1 = tf.keras.layers.Dropout(0.2)
        self.repeat_vector = tf.keras.layers.RepeatVector(n=50)

        # Decoder part
        self.lstm2 = tf.keras.layers.LSTM(units=50, activation='relu', return_sequences=True, unroll=True)
        self.conv2 = tf.keras.layers.Conv1D(filters=32, kernel_size=15, padding='same',
                                            data_format='channels_last', dilation_rate=1,
                                            activation='linear')
        self.dropout2 = tf.keras.layers.Dropout(0.2)

        # Final TimeDistributed Dense layer to output same number of features as input
        self.time_dense = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units=5))

    def call(self, inputs, training=False):
        # inputs: shape (batch, timesteps, features), e.g. (1, 50, 5)
        x = self.conv1(inputs)
        x = self.lstm1(x)
        x = self.dropout1(x, training=training)
        x = self.repeat_vector(x)
        x = self.lstm2(x)
        x = self.conv2(x)
        x = self.dropout2(x, training=training)
        x = self.time_dense(x)
        return x

def my_model_function():
    # Return an instance of MyModel.
    # Weights are not loaded here since the original model is loaded from disk in the issue,
    # but we only reconstruct the model architecture.
    return MyModel()

def GetInput():
    # Create a random tensor of shape (1, 50, 5) matching the input used in the example
    # Batch size 1, timesteps 50, feature dim 5
    return tf.random.uniform((1, 50, 5), dtype=tf.float32)

