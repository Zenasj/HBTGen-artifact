# tf.random.uniform((B,), dtype=tf.string) ← The Universal Sentence Encoder and similar models typically expect a batch of strings as input

import tensorflow as tf
import tensorflow_hub as hub

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Using TF Hub KerasLayer with Universal Sentence Encoder v4 as an example
        # Input is a batch of strings
        # NOTE: The original issue was about TFLite conversion problems with certain TF Hub models.
        # This model encapsulates the USE as a submodule.
        self.use_layer = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder/4", trainable=False)

    def call(self, inputs):
        # inputs: tf.Tensor of dtype tf.string, shape=(batch_size,)
        # Return the USE embedding vectors
        return self.use_layer(inputs)

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a batch of example strings as input to MyModel
    # Here, create a tf.Tensor of shape (batch_size,) of dtype string.
    batch_size = 2  # example batch size
    example_sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "TensorFlow is an open source machine learning framework."
    ]
    return tf.constant(example_sentences, dtype=tf.string)

