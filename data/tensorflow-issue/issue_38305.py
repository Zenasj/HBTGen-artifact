# tf.random.uniform((B, 1), dtype=tf.int64) ← The input is a batch of single integer IDs, shape (batch_size, 1)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Keys and corresponding mapped values for lookup
        keys = tf.constant([0, 1, 2, 3], dtype=tf.int64)
        values = tf.constant([0, 10, 20, 30], dtype=tf.int64)
        initializer = tf.lookup.KeyValueTensorInitializer(
            keys=keys,
            values=values,
            key_dtype=tf.int64,
            value_dtype=tf.int64,
        )
        # StaticHashTable created once and kept as a resource in the layer
        self.table = tf.lookup.StaticHashTable(initializer, default_value=-1)

        # Embedding layer for the output of the lookup operation
        self.embedder = tf.keras.layers.Embedding(
            input_dim=tf.reduce_max(values).numpy() + 1,
            output_dim=10,
        )

    def call(self, inputs):
        """
        inputs: tf.Tensor of shape (batch_size, 1), dtype tf.int64
        Returns:
          tf.Tensor of embeddings with shape (batch_size, 1, embedding_dim)
        """
        # Use the table to map input IDs to translated IDs
        translated = self.table.lookup(inputs)
        # Embed the translated IDs
        embedded = self.embedder(translated)
        return embedded

def my_model_function():
    # Return an instance of MyModel
    # The table is initialized internally in __init__, so nothing else to do
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the expected input:
    # Shape: (batch_size, 1), dtype=tf.int64, values in keys range (0 to 3)
    batch_size = 4  # Example batch size
    # Random integers between 0 and 3 inclusive
    inp = tf.random.uniform(shape=(batch_size, 1), minval=0, maxval=4, dtype=tf.int64)
    return inp

