# tf.random.uniform((B, 4), dtype=tf.int32) ← Based on input shape (4,) of integer IDs for embedding lookup

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, train_embeddings=True):
        super().__init__()
        # Embedding layer vocab_size=3, embedding_dim=2
        self.embedding = tf.keras.layers.Embedding(3, 2)
        # Set trainable according to constructor arg, to properly reflect in serialization
        self.embedding.trainable = train_embeddings
        self.dense = tf.keras.layers.Dense(4)

    def call(self, inputs, **kwargs):
        x = self.embedding(inputs)
        return self.dense(x)

    def get_config(self):
        # Include train_embeddings flag in config to preserve trainable state of sublayer
        config = super().get_config()
        config['train_embeddings'] = self.embedding.trainable
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def my_model_function():
    # By default create model with embedding.trainable == True
    return MyModel()

def GetInput():
    # Generate an input tensor matching the input expected by MyModel: shape (batch, 4), integer dtype (indices for embedding)
    # Assume batch size of 2 for demonstration
    batch_size = 2
    input_length = 4
    # Random integers between 0 and vocab_size-1 = 2
    return tf.random.uniform(
        (batch_size, input_length),
        minval=0,
        maxval=3,
        dtype=tf.int32
    )

