# tf.random.uniform((B, None), dtype=tf.string)  ← Input shape is (batch_size, variable_length_string_sequence)

import tensorflow as tf
import tensorflow_hub as hub

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # The TF Hub model expects input of shape (batch_size,) of dtype string
        # using the URL for the "gnews-swivel-20dim" embedding
        self.hub_layer = hub.KerasLayer(
            "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1",
            trainable=True)

        # Following the original example, 3 Conv1D layers and flatten + final sigmoid output
        # However, hub_layer output shape is (batch_size, embedding_dim) not sequence,
        # so Conv1D doesn't apply directly. We adjust with a 1-step temporal dim to apply Conv1D.

        # Add a dimension to match Conv1D input: (batch_size, sequence_length=1, channels=embedding_dim)
        # Then Conv1D with kernel size 3 with padding "same" is possible.
        self.expand_dims = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=1))

        self.conv1 = tf.keras.layers.Conv1D(64, 3, padding="same", activation="relu")
        self.conv2 = tf.keras.layers.Conv1D(32, 3, padding="same", activation="relu")
        self.conv3 = tf.keras.layers.Conv1D(16, 3, padding="same", activation="relu")
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(1, activation="sigmoid")

    def call(self, inputs):
        """
        The inputs are expected to be a 1-D tensor of dtype string, shape (batch_size,)
        representing raw text input strings (like sentences).

        The hub layer maps each string to a fixed 20-dimensional embedding vector.

        To apply Conv1D, we expand dims to (batch, 1, embedding_dim) and then apply Conv layers.
        """
        x = self.hub_layer(inputs)  # shape: (batch_size, 20)
        x = self.expand_dims(x)     # shape: (batch_size, 1, 20)
        x = self.conv1(x)           # (batch_size, 1, 64)
        x = self.conv2(x)           # (batch_size, 1, 32)
        x = self.conv3(x)           # (batch_size, 1, 16)
        x = self.flatten(x)         # (batch_size, 16)
        out = self.dense(x)         # (batch_size, 1) sigmoid output
        return out

def my_model_function():
    return MyModel()

def GetInput():
    """
    Returns a batch of string inputs compatible with MyModel.

    Since the hub embedding expects string inputs (e.g., sentences),
    the tensor returned must be a 1-D tf.string tensor.

    Here we provide a batch of 4 sample simple English sentences.
    """
    sample_sentences = [
        "This is a good movie.",
        "I did not like the film.",
        "It was an amazing experience.",
        "The plot was boring and predictable."
    ]
    return tf.constant(sample_sentences)

