# tf.random.uniform((B, 1), dtype=tf.float32) ← The example uses inputs of shape (1,) or (1,1), but the submodel input shape is (1,)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define a simple submodel with a single ReLU layer applied to input shape (1,)
        sub_in = tf.keras.Input(shape=(1,))
        self.relu_layer = tf.keras.layers.ReLU()
        sub_out = self.relu_layer(sub_in)
        self.submodel = tf.keras.Model(sub_in, sub_out)

    def call(self, inputs, training=None):
        # Forward pass through submodel
        return self.submodel(inputs)

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    # According to the discussion, inputs tested were shape (1,) or (1,1)
    # The submodel expects shape (1,), but testing with (1,1) was also done to highlight the issue.
    # We'll generate a batch of size 4 for demonstration with shape (4,1)
    batch_size = 4
    input_shape = (1,)
    return tf.random.uniform((batch_size,) + input_shape, dtype=tf.float32)

