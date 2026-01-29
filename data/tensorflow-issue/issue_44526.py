# tf.random.uniform((1, 12, 100), dtype=tf.float32) ← Input tensor shape inferred from keras Input(shape=[12, 100], batch_size=1)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define the same layers as original model
        # Input shape: (batch_size=1, 12, 100)
        self.dense1 = tf.keras.layers.Dense(20, activation='relu')
        self.batchnorm = tf.keras.layers.BatchNormalization()
        self.fc1 = tf.keras.layers.Dense(10, activation=None)
        self.lstm1 = tf.keras.layers.LSTM(20, return_sequences=True, stateful=True)
        self.lstm2 = tf.keras.layers.LSTM(20, return_sequences=True, stateful=True)

    def call(self, inputs, training=False):
        # Forward pass replicating the Keras Functional model
        x = self.dense1(inputs)
        x = self.batchnorm(x, training=training)
        fc1_out = self.fc1(x)
        gru1_out = self.lstm1(fc1_out)
        gru2_out = self.lstm2(gru1_out)
        # Return a tuple to match original outputs: (fc1, gru2)
        return fc1_out, gru2_out

def my_model_function():
    """
    Returns an instance of MyModel with stateful LSTM layers.
    Since the original model uses stateful=True, the batch size must be fixed to 1.
    """
    model = MyModel()

    # Build the model once with fixed input shape for correct weight initialization
    batch_size = 1
    time_steps = 12
    feature_dim = 100
    dummy_input = tf.zeros((batch_size, time_steps, feature_dim))
    # Must call once to create the weights
    model(dummy_input)

    return model

def GetInput():
    """
    Returns a random tensor input matching the model input shape: (1, 12, 100)
    The dtype is float32 as expected by Dense/LSTM layers.
    """
    return tf.random.uniform((1, 12, 100), dtype=tf.float32)

