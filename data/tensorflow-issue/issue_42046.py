# tf.random.uniform((B,), dtype=string) ← Model expects a 1D string tensor input (batch of sentences)

import tensorflow as tf
import tensorflow_hub as hub

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Use the TF Hub Universal Sentence Encoder KerasLayer (trainable)
        # This layer expects a 1D tensor of strings as input: shape (batch_size,)
        self.embed = hub.KerasLayer(
            "https://tfhub.dev/google/universal-sentence-encoder/4",
            output_shape=[512],
            input_shape=[],
            dtype=tf.string,
            trainable=True,
        )
        # Final regression layer
        self.dense = tf.keras.layers.Dense(1)

    def call(self, inputs, training=False):
        # Handle the implicit 2D expansion added by model.fit and Input layers:
        # inputs shape could be (batch_size, 1) instead of (batch_size,)
        # Squeeze last dimension if rank is 2 and last dim == 1
        if tf.rank(inputs) == 2 and inputs.shape[-1] == 1:
            inputs = tf.squeeze(inputs, axis=-1)

        x = self.embed(inputs)  # shape: (batch_size, 512)
        x = self.dense(x)       # shape: (batch_size, 1)
        return x

def my_model_function():
    # Return an instance of the model
    return MyModel()

def GetInput():
    # Return a 1D batch of string tensors simulating sentences
    # For example, batch size = 4, random sentences from some placeholder strings

    # Since inputs are strings, create a tf.constant of shape (batch_size,)
    batch_size = 4
    dummy_sentences = [
        "this is a test sentence",
        "another example sentence",
        "tensorflow hub universal sentence encoder",
        "keras model fit input issue",
    ]
    # Take batch_size samples (repeat or truncate as needed)
    inputs = tf.constant(dummy_sentences[:batch_size], dtype=tf.string)
    return inputs

