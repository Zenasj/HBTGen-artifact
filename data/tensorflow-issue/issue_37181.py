# tf.random.uniform((1000, 1), dtype=tf.int64)
import tensorflow as tf
from tensorflow.keras.layers import Embedding, Dense, Flatten, Input
from tensorflow.keras import Model

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Embedding layer with large input dimension (5,000,000) and embedding dim 16
        self.embed = Embedding(5000000, 16, name='embed')
        self.dense1 = Dense(10)
        self.dense2 = Dense(5)
        self.flatten = Flatten()
        self.output_dense = Dense(1, activation='sigmoid')
    
    def call(self, inputs):
        x = self.embed(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.flatten(x)
        return self.output_dense(x)

def my_model_function():
    # Returns an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random integer tensor of shape (1000, 1) with values in [0, 4999999]
    # Matches embedding input expectations (int64 indices)
    return tf.random.uniform((1000, 1), minval=0, maxval=5000000, dtype=tf.int64)

