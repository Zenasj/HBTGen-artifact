# tf.random.uniform((1, 100, 100), dtype=tf.float32) ← input shape from generator's output shape np.zeros((1, 100, 100))

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Build the model as per the example from the issue:
        # Input shape: (100, 100)
        # First flatten, then two outputs:
        #  - output_for_loss: Dense(2)
        #  - some_other_output: Reshape((2, -1)) from flattened
        self.flatten = tf.keras.layers.Flatten()
        self.output_for_loss = tf.keras.layers.Dense(2, name='output_for_loss')
        # To match the example's reshape, we need to infer the second dimension for reshape
        # Flatten input: shape=(batch, 100*100=10000)
        # Reshape target: (2, ?) means (?,) = 10000/2 = 5000
        self.some_other_output_reshape = tf.keras.layers.Reshape((2, 5000), name='some_other_output')

    def call(self, inputs, training=False):
        x = self.flatten(inputs)
        out1 = self.output_for_loss(x)
        out2 = self.some_other_output_reshape(x)
        return [out1, out2]

def my_model_function():
    # Return an instance of MyModel
    model = MyModel()
    # Compile with loss only for 'output_for_loss' output, same as in issue example
    model.compile(loss={'output_for_loss': tf.keras.losses.BinaryCrossentropy()})
    return model

def GetInput():
    # Return matching input for model: shape (batch=1, 100, 100), float32 tensor
    return tf.random.uniform((1, 100, 100), dtype=tf.float32)

