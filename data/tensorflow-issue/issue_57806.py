# tf.constant([B], dtype=tf.int64) ← input is a 1D tensor of int64 keys for vocabulary lookup

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # IMPORTANT NOTE:
        # The input asset file ("chzhu_vocab.txt") path must be handled carefully when saving/loading,
        # as described in the issue. To avoid asset vanishing on re-saving and loading,
        # we hold the initializer as a persistent attribute (self.initializer).
        #
        # The vocabulary file "chzhu_vocab.txt" is expected to exist alongside the saved model's asset folder.
        #
        # This example uses a StaticVocabularyTable with a TextFileInitializer.
        self.initializer = tf.lookup.TextFileInitializer(
            "chzhu_vocab.txt",  # Assumed to be in assets folder when saved
            key_dtype=tf.int64,
            key_index=0,
            value_dtype=tf.int64,
            value_index=1,
            delimiter=" ",
        )
        self.table = tf.lookup.StaticVocabularyTable(self.initializer, 1)

    @tf.function
    def call(self, inputs):
        # inputs: int64 tensor shape [B]
        return self.table.lookup(inputs)

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random batch of int64 keys that can be looked up in the vocabulary table.
    # Without the actual vocab file content, we guess keys from 0 to 10.
    # Batch size = 4 as example.
    batch_size = 4
    keys = tf.random.uniform(shape=(batch_size,), minval=0, maxval=10, dtype=tf.int64)
    return keys

