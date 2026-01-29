# tf.random.uniform((B, T, F), dtype=tf.float32)  # B=batch size, T=sequence length, F=feature size

import tensorflow as tf
from tensorflow import keras

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Bidirectional LSTM with 128 units
        self.bi_lstm = keras.layers.Bidirectional(
            keras.layers.LSTM(units=128), input_shape=(None, None)
        )
        self.dropout = keras.layers.Dropout(rate=0.5)
        self.dense1 = keras.layers.Dense(units=128, activation='relu')
        # Output layer with softmax activation
        # Assuming output size is y_train.shape[1]; here we use a common use case of 10 classes as placeholder
        self.dense2 = keras.layers.Dense(units=10, activation='softmax')

    def call(self, inputs, training=False):
        # inputs shape: (batch, time_steps, features)
        x = self.bi_lstm(inputs)
        if training:
            x = self.dropout(x, training=training)
        x = self.dense1(x)
        return self.dense2(x)

def my_model_function():
    # Return an instance of MyModel
    # Note: Without trained weights since none provided;
    # user is expected to train or load weights separately.
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the expected input shape.
    # From model code: input_shape=[x_train.shape[1], x_train.shape[2]]
    # We assume a batch size of 4, sequence length 100, and feature size 20 as plausible defaults.
    batch_size = 4
    sequence_length = 100
    feature_size = 20
    return tf.random.uniform((batch_size, sequence_length, feature_size), dtype=tf.float32)

