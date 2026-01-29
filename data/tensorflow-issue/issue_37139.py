# tf.random.uniform((B, ), dtype=tf.float32) for inputs: wide_input (shape [None,5])
# and tf.random.uniform((B, ), dtype=tf.float32) for inputs: deep_input (shape [None,6])

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Layers corresponding to the "deep" branch
        self.hidden1 = tf.keras.layers.Dense(30, activation="relu")
        self.hidden2 = tf.keras.layers.Dense(30, activation="relu")
        # Output layer
        self.output_layer = tf.keras.layers.Dense(1)
    
    def call(self, inputs):
        # Expecting inputs as a tuple/list of two tensors: (wide_input, deep_input)
        wide_input, deep_input = inputs
        
        # Pass deep_input through dense layers
        hidden1_out = self.hidden1(deep_input)
        hidden2_out = self.hidden2(hidden1_out)
        
        # Concatenate wide_input with processed deep branch output
        concat = tf.concat([wide_input, hidden2_out], axis=-1)
        
        # Final output layer
        output = self.output_layer(concat)
        return output

def my_model_function():
    # Return an instance of MyModel initialized per the reconstructed architecture
    return MyModel()

def GetInput():
    # Generate a valid input tuple of tensors that match model's input shapes ([None,5], [None,6])
    # Use batch size 2 for example
    batch_size = 2
    wide_input = tf.random.uniform((batch_size, 5), dtype=tf.float32)
    deep_input = tf.random.uniform((batch_size, 6), dtype=tf.float32)
    return (wide_input, deep_input)

