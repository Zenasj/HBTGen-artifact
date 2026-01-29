# tf.random.uniform((B,), dtype=tf.string) ← input is a 1D tf.string tensor with shape [None]

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # StaticHashTable mapping strings to integers:
        keys = tf.constant(['testing', 'this', 'thing'])
        values = tf.constant([1, 2, 3], dtype=tf.int32)
        initializer = tf.lookup.KeyValueTensorInitializer(keys, values)
        self._hash = tf.lookup.StaticHashTable(initializer, default_value=-1)

    @tf.function
    def call(self, word):
        # Perform lookup in the static hash table
        return self._hash.lookup(word)

def my_model_function():
    return MyModel()

def GetInput():
    # Return a 1D tf.string tensor input of shape [batch_size]
    # Using 3 elements as example to include some hits and misses.
    example_input = tf.constant(['testing', 'that', 'thing'], dtype=tf.string)
    return example_input

