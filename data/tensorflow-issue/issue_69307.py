# tf.random.uniform((B, 8), dtype=tf.float32), tf.random.uniform((B, 8), dtype=tf.float32)

import tensorflow as tf
from tensorflow import keras

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Two dense layers
        self.fc1 = keras.layers.Dense(8)
        self.fc2 = keras.layers.Dense(1)

    def build(self, input_shape):
        # input_shape here is a list/tuple of shapes for multiple inputs
        # sum second dimension sizes of all input shapes to set fc1 input size correctly
        total_input_dim = sum(shape[1] for shape in input_shape)
        self.fc1.build((None, total_input_dim))
        self.fc2.build((None, 8))
        super().build(input_shape)  # Mark built

    @tf.function(jit_compile=True)
    def call(self, inputs, **kwargs):
        # Assuming inputs is a tuple/list of tensors each with shape (batch, 8)
        # Concatenate them along last dimension (axis 1)
        concat_inputs = keras.ops.concatenate(inputs, axis=1)  
        x = self.fc1(concat_inputs)
        return self.fc2(x)

def my_model_function():
    # Return an instance of MyModel with no pretrained weights
    model = MyModel()
    # Build with dummy input shapes for two inputs of size 8 each
    model.build([(None, 8), (None, 8)])
    return model

def GetInput():
    # Return a tuple of two random tensors to match model input:
    # Each tensor shape (batch_size, 8), dtype float32 to match Dense input
    batch_size = 4  # arbitrary batch size
    input1 = tf.random.uniform((batch_size, 8), dtype=tf.float32)
    input2 = tf.random.uniform((batch_size, 8), dtype=tf.float32)
    return (input1, input2)

