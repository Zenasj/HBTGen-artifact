# tf.random.uniform((None, 8), dtype=tf.float32)  ← Input shape is (batch_size, 8), batch_size is dynamic (None)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Define layers as per the reported model
        self.dense1 = tf.keras.layers.Dense(30, activation='relu', name='dense')
        self.dense2 = tf.keras.layers.Dense(30, activation='relu', name='dense_1')
        self.concat = tf.keras.layers.Concatenate(name='concatenate')
        self.output_dense = tf.keras.layers.Dense(1, name='dense_2')

    def call(self, inputs):
        # Forward pass mimics the functional API from the issue
        # inputs shape: (batch_size, 8)
        x1 = inputs
        x2 = self.dense1(x1)            # first dense layer on input
        x3 = self.dense2(x2)            # second dense layer on previous dense output
        x4 = self.concat([x1, x3])      # concatenate input and second dense output
        output = self.output_dense(x4)  # final dense layer producing output
        return output

def my_model_function():
    # Return an instance of the model
    return MyModel()

def GetInput():
    # Return a random tensor input matching the input shape (batch_size, 8)
    # Choosing batch size = 4 for example; dtype float32 is typical for TF models
    return tf.random.uniform((4, 8), dtype=tf.float32)

