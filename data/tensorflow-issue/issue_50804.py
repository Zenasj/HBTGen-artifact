# tf.random.uniform((2, 2, 2), dtype=tf.float32) ← inferred input shape from original dataset (samples, timesteps, features)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # LSTM with units equal to feature size (2)
        self.lstm = tf.keras.layers.LSTM(
            units=2,
            return_sequences=False,
            return_state=False,
            name='lstm',
        )
        self.dense = tf.keras.layers.Dense(
            units=1,
            name='dense',
        )
    
    def call(self, inputs, training=False):
        x = self.lstm(inputs)
        x = self.dense(x)
        return x

def my_model_function():
    # Return an instantiated and compiled MyModel
    model = MyModel()
    model.compile(
        optimizer='rmsprop',
        loss='mse',
        metrics=['mse'],
    )
    return model

def GetInput():
    # Return a random input tensor shaped (batch=1, timesteps=2, features=2)
    # This matches the expected input shape of the model's LSTM layer.
    return tf.random.uniform((1, 2, 2), dtype=tf.float32)

