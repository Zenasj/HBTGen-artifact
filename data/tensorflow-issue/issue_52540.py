# tf.random.uniform((B=1, H=200), dtype=tf.int32) ← Input shape inferred from keras.Input(shape=(200,), dtype="int32")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Assumptions:
        # - max_features is not specified in the original snippet.
        #   We set an example vocabulary size 200 here to match input range.
        self.max_features = 200  
        self.embedding_dim = 256
        self.lstm_units = 64
        
        self.embedding = layers.Embedding(self.max_features, self.embedding_dim)
        self.lstm = layers.LSTM(self.lstm_units)
        self.classifier = layers.Dense(1, activation="sigmoid")
        
    def call(self, inputs, training=False):
        # inputs: (batch_size, 200) int32 tensor
        x = self.embedding(inputs)  # (batch, 200, 256)
        x = self.lstm(x)            # (batch, 64)
        outputs = self.classifier(x)  # (batch, 1)
        return outputs

def my_model_function():
    # Return a fresh instance of MyModel
    return MyModel()

def GetInput():
    # Return a random integer tensor matching input expected by MyModel
    # Input shape: (batch_size=1, sequence_length=200)
    # Values in [1, 199] reflecting input range in build_fn from original code
    # dtype: int32 as required by Embedding layer
    return tf.random.uniform((1, 200), minval=1, maxval=200, dtype=tf.int32)

