import random
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf

class TestModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.gru = tf.keras.layers.GRU(10, return_sequences=True, return_state=True)

    def call(self, inputs):
        return self.gru(inputs)

# Create and test the model
model = TestModel()
test_input = tf.random.uniform((2, 3, 5))  # Batch size = 2, sequence length = 3, feature size = 5
output = model(test_input)
print("Output types and shapes:", [(type(o), o.shape) for o in output])