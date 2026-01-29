# tf.random.uniform((B,), dtype=tf.string) ← Input shape is batch of strings (1D tensor of strings)

import tensorflow as tf
import tensorflow_hub as hub

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Using the Universal Sentence Encoder Multilingual from TF Hub
        # The original model sets trainable=True on KerasLayer
        # Inputs are strings batches, output embeddings are float32 vectors
        self.embed = hub.KerasLayer(
            "https://tfhub.dev/google/universal-sentence-encoder-multilingual/3",
            dtype=tf.string,
            trainable=True,
        )

    def call(self, inputs):
        # Expecting inputs as a tuple or list of two string tensors: (s1, s2)
        s1, s2 = inputs

        # Get embeddings for s1 and s2: each shape (batch_size, embedding_dim)
        v1 = self.embed(s1)
        v2 = self.embed(s2)

        # Compute a kind of similarity measure by reduced sum of element-wise multiplication
        # Resulting shape: (batch_size,)
        cd = tf.reduce_sum(v1 * v2, axis=-1)
        return cd


def my_model_function():
    # Return an instance of MyModel
    return MyModel()


def GetInput():
    # Return a tuple of two 1D string tensors to serve as compatible input to MyModel
    # For example, 3 strings each, batch size = 3
    s1 = tf.constant(["x", "y", "z"])
    s2 = tf.constant(["a", "b", "c"])
    return (s1, s2)

