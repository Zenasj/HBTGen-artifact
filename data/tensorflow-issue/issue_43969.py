# tf.random.uniform((B,), dtype=tf.float32) ← input is a 1D float tensor, e.g. [1, 2, 3]

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # No trainable parameters; the model just replicates input in a tuple
        # This mimics the dataset.map(lambda x: (x, x)) behavior from the issue
        
    @tf.function(jit_compile=True)
    def call(self, inputs):
        # inputs is a tensor (e.g. shape (B,))
        # Return a tuple of inputs duplicated, as in dataset.map(lambda x: (x, x))
        # This matches the mapping function from the issue
        return (inputs, inputs)

def my_model_function():
    # Return instance of MyModel
    return MyModel()

def GetInput():
    # Generate input tensor matching expected input: 1D float tensor with some batch size
    # The original issue used tf.constant([1, 2, 3], dtype=tf.float32)
    # We'll generate uniform float tensor of shape (3,)
    return tf.random.uniform((3,), dtype=tf.float32)

