# tf.random.uniform((B, 4), dtype=tf.float32) ← Input shape is inferred from train_x of shape (100, 4)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, hidden_units):
        super().__init__()
        # Single hidden dense layer with ReLU activation and output layer with logits for 3 classes
        self.hidden_layer = tf.keras.layers.Dense(hidden_units, activation='relu')
        self.output_layer = tf.keras.layers.Dense(3)  # logits for 3 classes

    def call(self, inputs, training=False):
        x = self.hidden_layer(inputs)
        logits = self.output_layer(x)
        return logits


def my_model_function(hidden_units=10):
    """
    Return an instance of MyModel initialized with hidden_units.
    Default hidden_units=10 to reflect the initial usage in the original example.
    """
    return MyModel(hidden_units=hidden_units)


def GetInput():
    """
    Return a random tensor input matching the expected input shape of MyModel.
    According to the original example train_x has shape (100, 4), but the batch size is flexible.
    Here we generate a random batch of size 32.
    """
    batch_size = 32  # arbitrary batch size
    input_dim = 4    # features dimension, matching original example
    
    # Generate uniform random inputs as float32 tensor matching input shape
    return tf.random.uniform((batch_size, input_dim), dtype=tf.float32)

