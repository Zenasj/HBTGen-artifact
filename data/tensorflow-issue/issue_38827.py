# tf.random.uniform((B, 60, 50), dtype=tf.float32) ← Inferred input shape: batch size B, sequence length 60, feature size 50 (51 columns total minus last column label)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # LSTM layers with BatchNormalization and Dropout
        self.lstm1 = tf.keras.layers.LSTM(128, return_sequences=True, input_shape=(60, 50))
        self.dropout1 = tf.keras.layers.Dropout(0.2)
        self.batchnorm1 = tf.keras.layers.BatchNormalization()

        self.lstm2 = tf.keras.layers.LSTM(128, return_sequences=True)
        self.dropout2 = tf.keras.layers.Dropout(0.2)
        self.batchnorm2 = tf.keras.layers.BatchNormalization()

        self.lstm3 = tf.keras.layers.LSTM(128, return_sequences=True)
        self.dropout3 = tf.keras.layers.Dropout(0.2)
        self.batchnorm3 = tf.keras.layers.BatchNormalization()

        self.lstm4 = tf.keras.layers.LSTM(128, return_sequences=False)
        self.dropout4 = tf.keras.layers.Dropout(0.5)
        self.batchnorm4 = tf.keras.layers.BatchNormalization()

        self.dense = tf.keras.layers.Dense(1)

    def call(self, inputs, training=False):
        x = self.lstm1(inputs)
        x = self.dropout1(x, training=training)
        x = self.batchnorm1(x, training=training)

        x = self.lstm2(x)
        x = self.dropout2(x, training=training)
        x = self.batchnorm2(x, training=training)

        x = self.lstm3(x)
        x = self.dropout3(x, training=training)
        x = self.batchnorm3(x, training=training)

        x = self.lstm4(x)
        x = self.dropout4(x, training=training)
        x = self.batchnorm4(x, training=training)

        output = self.dense(x)
        return output


def my_model_function():
    # Return an instance of MyModel
    model = MyModel()
    # Compile the model with same loss, optimizer, and metric as original snippet
    model.compile(
        loss='mean_squared_error',
        optimizer='adam',
        metrics=['mean_absolute_error']
    )
    return model


def GetInput():
    # Generate random input tensor shaped as (batch_size, sequence_length, features)
    # From original code: sequence length = 60, features = 50 (51 columns minus last label column)
    batch_size = 32  # chosen batch size matching original code
    seq_len = 60
    features = 50
    # Use float32 dtype as standard for TensorFlow models
    return tf.random.uniform((batch_size, seq_len, features), dtype=tf.float32)

