# tf.random.uniform((B, 128, 3072), dtype=tf.float32) ← Input shape inferred from provided Keras example inputs (batch size flexible)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Bidirectional LSTM layer with 768 units each direction (default activation tanh)
        self.bilstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(units=768, activation='tanh'),
            name='Bidirectional_LSTM'
        )
        # Layers for combining outputs
        self.subtract = tf.keras.layers.Subtract(name='Subtract')
        self.abs_lambda = tf.keras.layers.Lambda(lambda x: tf.abs(x), name='Abs')
        self.multiply = tf.keras.layers.Multiply(name='Multiply')
        self.concat = tf.keras.layers.Concatenate(name='Concat')
        self.dense = tf.keras.layers.Dense(units=1, name='Output_Dense')

    def call(self, inputs, training=None):
        # inputs is a tuple/list: (left, right)
        left, right = inputs
        l_lstm = self.bilstm(left)
        r_lstm = self.bilstm(right)

        subtracted = self.subtract([l_lstm, r_lstm])
        abs_subtracted = self.abs_lambda(subtracted)
        mul = self.multiply([l_lstm, r_lstm])
        concat = self.concat([abs_subtracted, mul])

        output = self.dense(concat)
        return output

def my_model_function():
    # Instantiate MyModel; weights initialized randomly.
    return MyModel()

def GetInput():
    # Return tuple of two random inputs compatible with model inputs:
    # shape (batch_size, 128, 3072), dtype float32.
    # batch_size chosen as 4 here for demo; can be different.
    batch_size = 4
    left_input = tf.random.uniform(shape=(batch_size, 128, 3072), dtype=tf.float32)
    right_input = tf.random.uniform(shape=(batch_size, 128, 3072), dtype=tf.float32)
    return (left_input, right_input)

