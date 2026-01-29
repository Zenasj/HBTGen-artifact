# tf.random.uniform((B, 1), dtype=tf.int64) ← input shape is (batch_size, 1) with int64 dtype

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Initialize the lookup table with predefined keys and string values
        # Keys are int64: [0,1,2], values are strings: ["A", "B", "C"]
        table_init = tf.lookup.KeyValueTensorInitializer(
            key_dtype=tf.int64,
            keys=[0, 1, 2],
            value_dtype=tf.string,
            values=["A", "B", "C"],
            name="table_init")
        self.index_to_kw = tf.lookup.StaticHashTable(table_init, default_value="?")

    def call(self, inputs, **kwargs):
        # Perform lookup on inputs, returning string tensor
        return self.index_to_kw.lookup(inputs)


def my_model_function():
    # Returns an instance of MyModel.
    # No weights loading required since StaticHashTable is initialized internally.
    return MyModel()


def GetInput():
    # Returns a random input tensor compatible with MyModel:
    # Example shape (batch_size=3, 1), dtype int64 with values in [0..2] (valid keys)
    batch_size = 3
    # Inputs are int64 integers between 0 and 2, shape (batch_size, 1)
    return tf.random.uniform((batch_size, 1), minval=0, maxval=3, dtype=tf.int64)

