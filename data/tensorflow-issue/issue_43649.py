# tf.random.uniform((B, 1000, 100), dtype=tf.float32)  # Input shape inferred from model build (batch, time_steps, features)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Following the structure given in the issue's model:
        # Input: (None, 1000, 100)
        # Conv1D: 128 filters, kernel_size=7, strides=4, no activation here
        # BatchNorm + ReLU + Dropout
        # LSTM 64 units (return_sequences=True) + Dropout + BatchNorm
        # LSTM 64 units (return_sequences=True) + Dropout + BatchNorm + Dropout
        # TimeDistributed Dense(1, activation='sigmoid')
        
        self.conv1d = tf.keras.layers.Conv1D(128, 7, strides=4, padding="valid")
        self.bn0 = tf.keras.layers.BatchNormalization()
        self.act = tf.keras.layers.Activation("relu")
        self.dropout0 = tf.keras.layers.Dropout(0.5)
        
        self.lstm1 = tf.keras.layers.LSTM(64, return_sequences=True)
        self.dropout1 = tf.keras.layers.Dropout(0.5)
        self.bn1 = tf.keras.layers.BatchNormalization()
        
        self.lstm2 = tf.keras.layers.LSTM(64, return_sequences=True)
        self.dropout2 = tf.keras.layers.Dropout(0.5)
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.dropout3 = tf.keras.layers.Dropout(0.5)
        
        self.time_dense = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(1, activation="sigmoid")
        )
    
    def call(self, inputs, training=None):
        x = self.conv1d(inputs)
        x = self.bn0(x, training=training)
        x = self.act(x)
        x = self.dropout0(x, training=training)
        
        x = self.lstm1(x)
        x = self.dropout1(x, training=training)
        x = self.bn1(x, training=training)
        
        x = self.lstm2(x)
        x = self.dropout2(x, training=training)
        x = self.bn2(x, training=training)
        x = self.dropout3(x, training=training)
        
        x = self.time_dense(x)
        return x


def my_model_function():
    # Build and return an instance of MyModel.
    # No weights are restored here to keep consistent with provided info.
    return MyModel()

def GetInput():
    # Return a tensor matching the model's input: batch size 10 (arbitrary here),
    # time steps = 1000, features = 100, dtype float32
    # Use uniform distribution over [0,1) as example.
    return tf.random.uniform((10, 1000, 100), dtype=tf.float32)

