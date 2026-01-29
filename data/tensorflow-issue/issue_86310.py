# tf.random.uniform((32, 5, 1), dtype=tf.float32) ← inferred input shape based on batch_size=32, sequence_length=5, num_features=1

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Stateful LSTM expects fixed batch size and input shape
        # With Keras 3 / TF 2.18+, batch_input_shape is deprecated in LSTM layer directly.
        # Instead, use an explicit Input layer with batch_shape, then pass to LSTM with stateful=True.
        batch_size = 32
        sequence_length = 5
        num_features = 1
        
        # Define an Input layer with batch_shape to fix batch size and sequence length
        self.input_layer = tf.keras.layers.InputLayer(batch_shape=(batch_size, sequence_length, num_features))
        
        self.lstm = tf.keras.layers.LSTM(units=64, stateful=True, return_sequences=False)
        self.dense = tf.keras.layers.Dense(units=1)
        
    def call(self, inputs, training=None):
        x = self.input_layer(inputs)
        x = self.lstm(x, training=training)
        output = self.dense(x)
        return output
    
    def reset_states(self):
        # Reset states of the internal lstm cell, useful after each epoch in training
        self.lstm.reset_states()


def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return random input tensor matching fixed batch size, sequence length and feature dimension
    batch_size = 32
    sequence_length = 5
    num_features = 1
    # Using float32, uniform distribution as example input data
    return tf.random.uniform((batch_size, sequence_length, num_features), dtype=tf.float32)

