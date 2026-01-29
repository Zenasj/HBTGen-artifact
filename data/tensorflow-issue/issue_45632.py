# tf.random.uniform((B, 2, 5 or 6), dtype=tf.float32) ← Input shapes: two inputs [5], [6]

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Layers corresponding to the "deep" input processing
        self.dense1 = tf.keras.layers.Dense(30, activation='relu')
        self.dense2 = tf.keras.layers.Dense(30, activation='relu')
        # Output layer after concatenation
        self.output_layer = tf.keras.layers.Dense(1, name="output")
    
    def call(self, inputs, training=False):
        # Expecting inputs as a tuple/list of two tensors: (input_A, input_B)
        input_A, input_B = inputs  # input_A shape: [batch_size, 5], input_B shape: [batch_size, 6]
        
        hidden1 = self.dense1(input_B)
        hidden2 = self.dense2(hidden1)
        concat = tf.concat([input_A, hidden2], axis=-1)  # Concatenate on last axis
        output = self.output_layer(concat)
        return output

def my_model_function():
    # Return an instance of the model
    return MyModel()

def GetInput():
    # Generate a tuple of two random tensors matching the model inputs:
    # input_A: shape (batch_size, 5)
    # input_B: shape (batch_size, 6)
    # For example, batch size = 4 (arbitrary choice)
    batch_size = 4
    input_A = tf.random.uniform((batch_size, 5), dtype=tf.float32)
    input_B = tf.random.uniform((batch_size, 6), dtype=tf.float32)
    return (input_A, input_B)

