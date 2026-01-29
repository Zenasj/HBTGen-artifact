# tf.random.uniform((B, 1), dtype=tf.string) ← Based on input shape 1 scalar string per batch item

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Use StringLookup layer with vocabulary ["a", "b"] as in the issue repro
        # Use the current recommended import path for StringLookup in TF 2.20
        # (The issue references tf.keras.layers.experimental.preprocessing.StringLookup,
        # in recent TF versions it's tf.keras.layers.StringLookup)
        self.lookup = tf.keras.layers.StringLookup(vocabulary=["a", "b"])

    def call(self, inputs):
        # inputs expected to be shape (B, 1), dtype string
        return self.lookup(inputs)

def my_model_function():
    # Returns an instance of MyModel with default initialization
    return MyModel()

def GetInput():
    # Return an input tensor matching expected input shape and dtype
    # The input shape is (batch_size, 1), with dtype string, 
    # strings chosen randomly from the vocabulary ["a", "b"] for demonstration.
    # Using tf.random.uniform is not possible for dtype string,
    # so we create a string tensor with random choice from the vocabulary via tf.random.uniform for indices.
    vocab = tf.constant(["a", "b"])
    batch_size = 4  # arbitrary batch size for input generation
    random_indices = tf.random.uniform(shape=(batch_size, 1), minval=0, maxval=2, dtype=tf.int32)
    inputs = tf.gather(vocab, random_indices)
    # tf.gather returns shape (batch_size, 1), dtype string as required
    return inputs

