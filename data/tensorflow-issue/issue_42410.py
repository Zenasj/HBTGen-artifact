# tf.random.uniform((1, None), dtype=tf.int32) ← Input shape is [batch_size=1, sequence_length=variable] with int32 indices

import tensorflow as tf

class Embed(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

    def build(self, input_shape):
        self.embeddings = self.add_weight(
            "weight",
            shape=[self.vocab_size, self.embed_dim],
            initializer=tf.keras.initializers.GlorotNormal(),
            trainable=True,
        )

    def call(self, inputs):
        # Cast inputs to int32 for indexing safety
        indices = tf.cast(inputs, tf.int32)
        # Use tf.gather to lookup embeddings along vocab dimension
        return tf.gather(self.embeddings, indices)

class MyModel(tf.keras.Model):
    def __init__(self, vocab_size=29, embed_dim=320, **kwargs):
        super().__init__(**kwargs)
        self.embed = Embed(vocab_size, embed_dim)
        self.dense = tf.keras.layers.Dense(350)

    def call(self, inputs, training=False):
        # Embed input tokens
        x = self.embed(inputs)
        # Apply a fully connected layer
        output = self.dense(x)
        return output


def my_model_function():
    model = MyModel()
    # Build the model by calling on a sample input with shape [1, None]
    model(tf.keras.Input(shape=(None,), dtype=tf.int32))
    return model

def GetInput():
    # Return a random tensor input that matches expected input shape and type
    # Shape: batch=1, sequence_length=100 (arbitrary fixed length for example)
    # Values between 0 and vocab_size-1 (here 29)
    vocab_size = 29
    sequence_length = 100
    return tf.random.uniform(
        shape=(1, sequence_length),
        minval=0,
        maxval=vocab_size,
        dtype=tf.int32
    )

