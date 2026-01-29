# tf.random.uniform((B, 2), dtype=tf.float32) ← The model expects input shape (batch_size, 2)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Reconstruct the model from the issue's example:
        # - 3 Dense layers with relu for first two, no activation for last
        # - First layer input shape 2
        self.dense1 = tf.keras.layers.Dense(2, activation='relu')  
        self.dense2 = tf.keras.layers.Dense(2, activation='relu')
        self.dense3 = tf.keras.layers.Dense(1)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        return x

def my_model_function():
    # Create and return an instance of MyModel;
    # weights are uninitialized here, since the issue shows training is needed before usage.
    # Weights can be loaded/assigned externally if desired.
    return MyModel()

def GetInput():
    # Return a random batch of input matching input shape (batch_size, 2)
    # Based on the issue, inputs are 2D integer values in range ~0 to 50000.
    # The original model input was fed numpy int arrays. We'll generate float32 tensors,
    # since Keras typically expects float inputs. Alternatively, cast int to float.
    import numpy as np
    batch_size = 10
    # Generate 2 int features per sample similar to issue inputs:
    np_input = np.random.randint(0, 50000, size=(batch_size, 2)).astype('float32')
    return tf.convert_to_tensor(np_input)

