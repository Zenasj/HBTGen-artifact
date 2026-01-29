# tf.random.uniform((1, 2), dtype=tf.int32) ← Based on input_signature shape from the original examples

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Use a Keras Embedding layer similar to original case 2 example
        # Embedding with vocab size 3 and embedding dim 4 as per original code
        self.embedding = tf.keras.layers.Embedding(input_dim=3, output_dim=4)

    @tf.function(input_signature=[tf.TensorSpec(shape=[1, 2], dtype=tf.int32)])
    def call(self, inputs):
        # Forward call through embedding layer
        return self.embedding(inputs)

def my_model_function():
    # Return an instance of MyModel; no special initialization needed beyond constructor
    return MyModel()

def GetInput():
    # Return a random int32 tensor of shape (1, 2) in [0, 3) as valid indices for embedding
    return tf.random.uniform(shape=(1, 2), minval=0, maxval=3, dtype=tf.int32)

