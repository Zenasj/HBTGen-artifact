# tf.random.uniform((B, None), dtype=tf.float32) ← input shape guessed as (batch, sequence_length) with 2 inputs, each (batch, None)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Two separate dense layers for each input
        self.hidden_layer_0 = tf.keras.layers.Dense(100, activation=tf.nn.relu)
        self.hidden_layer_1 = tf.keras.layers.Dense(100, activation=tf.nn.relu)
        self.concat = tf.keras.layers.Concatenate()
        self.out_layer = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)

    def call(self, inputs):
        """
        inputs: dict with keys 'input_0' and 'input_1'
        Each input tensor shape: (batch_size, sequence_length) with sequence_length = None (variable)
        """
        activation_0 = self.hidden_layer_0(inputs['input_0'])
        activation_1 = self.hidden_layer_1(inputs['input_1'])
        concat = self.concat([activation_0, activation_1])
        return self.out_layer(concat)

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Generate random inputs matching the input expected by MyModel.
    # Assumptions:
    # - Batch size = 32
    # - Sequence length = 10 (arbitrarily chosen fixed length for testing)
    # - Inputs are float32 tensors shaped (batch, seq_len)
    batch_size = 32
    seq_len = 10
    input_0 = tf.random.uniform((batch_size, seq_len), dtype=tf.float32)
    input_1 = tf.random.uniform((batch_size, seq_len), dtype=tf.float32)
    return {'input_0': input_0, 'input_1': input_1}

