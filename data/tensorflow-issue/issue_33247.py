# tf.random.uniform((B, 784, 3), dtype=tf.float32) ← Input shape inferred from keras.Input(shape=(784,3))

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # GRU layer with 64 units, relu activation, and dropout 0.1 as per original example
        self.gru = tf.keras.layers.GRU(64, activation='relu', dropout=0.1, name='GRU')
        # Dense layer with 64 units and relu activation
        self.dense = tf.keras.layers.Dense(64, activation='relu', name='dense')
        # Output Dense layer with 10 units and softmax activation
        self.predictions = tf.keras.layers.Dense(10, activation='softmax', name='predictions')

    def call(self, inputs, training=False):
        # Pass inputs through GRU layer (training flag needed for dropout)
        x = self.gru(inputs, training=training)
        x = self.dense(x)
        return self.predictions(x)

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random float tensor matching the input expected by MyModel: (batch_size=2, timesteps=784, features=3)
    # Batch size chosen arbitrarily as 2 (can be any positive integer)
    return tf.random.uniform((2, 784, 3), dtype=tf.float32)

