# tf.random.uniform((B, 10), dtype=tf.float32) ← Input shape (batch_size, num_features), inferred from 9 numeric features plus label

import tensorflow as tf
import functools
import numpy as np

# Assumptions:
# - Input is a batch of numeric features: ['Datum','Uhrzeit','Wochentag','Wochenende','Ferien','Feiertag','Brueckentag','Schneechaos','Streik']
# - Input shape is (batch_size, 9)
# - The model normalizes input based on stored mean and std.
# - Output layer originally was sigmoid with 1 output (seems binary classification)
# - Some original data handling code mapped features dict to a stacked tensor; here we assume direct float32 tensor input.

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Constants derived from dataset stats (placeholders for example).
        # In practice, these values would be replaced with dataset mean/std values
        # Here, we initialize with zeros and ones to avoid runtime errors without the data
        self.MEAN = tf.constant([0., 0., 0., 0., 0., 0., 0., 0., 0.], dtype=tf.float32)  # mean of numeric features
        self.STD = tf.constant([1., 1., 1., 1., 1., 1., 1., 1., 1.], dtype=tf.float32)   # std deviation of numeric features
        
        # Preprocessing layer that normalizes input features
        def normalize_numeric_data(data, mean, std):
            return (data - mean) / std

        self.normalize = functools.partial(normalize_numeric_data, mean=self.MEAN, std=self.STD)

        # Model layers following normalization
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(128, activation='relu')
        self.output_layer = tf.keras.layers.Dense(1, activation='sigmoid')
        
    def call(self, inputs, training=False):
        # inputs: tensor of shape (batch_size, 9) float32
        x = tf.cast(inputs, tf.float32)
        # Normalize features
        mean = tf.reshape(self.MEAN, [1, -1])
        std = tf.reshape(self.STD, [1, -1])
        x = (x - mean) / std
        # Forward pass
        x = self.dense1(x)
        x = self.dense2(x)
        output = self.output_layer(x)
        return output

def my_model_function():
    model = MyModel()
    # Optionally set MEAN and STD here if known:
    # Example values from issue (they were computed from a dataset.describe):
    # MEAN = np.array([...])
    # STD = np.array([...])
    # model.MEAN.assign(MEAN)
    # model.STD.assign(STD)
    return model

def GetInput():
    # Return a random input tensor matching the model's expected input shape:
    # batch dimension can be arbitrary, here set to 32
    batch_size = 32
    num_features = 9  # numeric features count
    # Random uniform inputs in a plausible range (e.g. between 0 and 1000) as original numeric features varied widely (e.g. dates, times)
    input_tensor = tf.random.uniform((batch_size, num_features), minval=0, maxval=1000, dtype=tf.float32)
    return input_tensor

