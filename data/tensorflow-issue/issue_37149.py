# tf.random.uniform((B, 20, 1), dtype=tf.float32) ← inferred input shape from window size 20 and feature dimension 1

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Bidirectional LSTM layer with 64 units
        self.bilstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64))
        # Dense layers for regression task
        self.dense1 = tf.keras.layers.Dense(25, activation='relu')
        self.dense2 = tf.keras.layers.Dense(1, activation='sigmoid')  # sigmoid used as per original

    def call(self, inputs, training=False):
        # Forward pass through the layers
        x = self.bilstm(inputs)
        x = self.dense1(x)
        output = self.dense2(x)
        return output

def my_model_function():
    # Return an instance of MyModel, compiled as in original snippet
    model = MyModel()
    model.compile(
        loss='mean_squared_error',
        optimizer=tf.keras.optimizers.Adam(1e-4),
        metrics=['accuracy']
    )
    return model

def GetInput():
    # Return a random input tensor matching the LSTM expected input shape:
    # Shape: (batch_size, time_steps=20, features=1)
    # Batch size chosen as 32 by default (common minibatch size)
    return tf.random.uniform((32, 20, 1), dtype=tf.float32)

