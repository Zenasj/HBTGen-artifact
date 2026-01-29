# tf.random.uniform((B, 1), dtype=tf.float32) ← Input shape is assumed (batch_size, 1) of float32 matching the example input [[0.5]]

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Initialize the StaticHashTable with key-value pairs "a":0.5, "b":0.7, default 0.0
        keys = tf.constant(["a", "b"], dtype=tf.string)
        values = tf.constant([0.5, 0.7], dtype=tf.float32)
        initializer = tf.lookup.KeyValueTensorInitializer(keys, values)
        # Important: store the lookup table as an attribute so it is tracked
        self.lookup_table = tf.lookup.StaticHashTable(initializer, default_value=0.0)
        # A simple Dense layer as the main model
        self.dense = tf.keras.layers.Dense(1)

    @tf.function
    def call(self, x):
        # x is a batch of string keys that we lookup in the table
        # lookup_table.lookup expects string tensor, output is float32 tensor
        x = self.lookup_table.lookup(x)
        # Dense expects float tensor of shape (batch, features)
        return self.dense(x)

def my_model_function():
    # Return an instance of MyModel
    # The model internally defines and tracks the StaticHashTable and a Dense layer.
    return MyModel()

def GetInput():
    # The MyModel expects a tf.string tensor input representing keys for lookup
    # We return a batch with a single example "a" to match key space
    return tf.constant([["a"]], dtype=tf.string)

