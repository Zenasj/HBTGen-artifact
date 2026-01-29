# tf.random.uniform((B, 200, 6), dtype=tf.float32)  # Inferred input shape from model input_shape=(200,6)

import tensorflow as tf
from tensorflow.keras import layers

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # LSTM layer with 128 units, returns sequences (output shape: (batch, 200, 128))
        self.lstm = layers.LSTM(128, return_sequences=True, name='lstm_1')
        # TimeDistributed Dense layer with 64 units and ReLU activation (applied to each time step)
        self.time_distributed = layers.TimeDistributed(layers.Dense(64, activation='relu'), name='time_distributed')
        # Flatten layer to collapse (batch, time_steps * features)
        self.flatten = layers.Flatten(name='flatten')
        # Dense layer with 64 units and ReLU activation
        self.dense1 = layers.Dense(64, activation='relu', name='dense_1')
        # Output Dense layer with 2 units and softmax activation for classification
        self.output_layer = layers.Dense(2, activation='softmax', name='output')
    
    def call(self, inputs, training=False):
        x = self.lstm(inputs)
        x = self.time_distributed(x)
        x = self.flatten(x)
        x = self.dense1(x)
        return self.output_layer(x)

def my_model_function():
    # Return an instance of MyModel with weights uninitialized (weights can be loaded if available)
    return MyModel()

def GetInput():
    # Return a random batch input tensor matching the model's input shape: (batch_size, time_steps=200, features=6)
    # Batch size is arbitrary, using 4 here
    batch_size = 4
    return tf.random.uniform((batch_size, 200, 6), dtype=tf.float32)

