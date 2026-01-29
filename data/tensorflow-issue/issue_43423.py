# tf.random.uniform((B, 3))  ← Based on the example input shape [None, 3] given for model inputs

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Two simple submodels both taking inputs of shape (None, 3) and outputting a scalar per example
        self.model1 = tf.keras.Sequential([
            tf.keras.layers.Dense(1)
        ])
        self.model2 = tf.keras.Sequential([
            tf.keras.layers.Dense(1)
        ])
        # Fusion layer concatenates outputs of model1 and model2, followed by tanh activated dense layer
        self.concat = tf.keras.layers.Concatenate()
        self.final_dense = tf.keras.layers.Dense(1, activation='tanh')

        # Build models explicitly with input shape (None, 3)
        self.model1.build(input_shape=(None, 3))
        self.model2.build(input_shape=(None, 3))

    def call(self, inputs, training=False):
        # inputs is expected to be a list or tuple of two tensors: [input_to_model1, input_to_model2]
        x1, x2 = inputs
        out1 = self.model1(x1)
        out2 = self.model2(x2)
        fusion = self.concat([out1, out2])
        out = self.final_dense(fusion)
        return out

def my_model_function():
    # Return an instance of MyModel (weights uninitialized randomly)
    return MyModel()

def GetInput():
    # Provide matching random input tensors for model1 and model2 inputs - shape (batch_size, 3)
    # We use batch_size=4 as a small example
    batch_size = 4
    input1 = tf.random.uniform((batch_size, 3), dtype=tf.float32)
    input2 = tf.random.uniform((batch_size, 3), dtype=tf.float32)
    return [input1, input2]

