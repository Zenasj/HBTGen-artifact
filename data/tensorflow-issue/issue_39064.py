# tf.random.uniform((B,), dtype=tf.string) ← Because lookup expects a 1D string tensor as input

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, vocab_file, num_oov=1):
        super().__init__()
        # Create the TextFileInitializer for lookup table initialization
        self.init = tf.lookup.TextFileInitializer(
            filename=vocab_file,
            key_dtype=tf.string,
            key_index=tf.lookup.TextFileIndex.WHOLE_LINE,
            value_dtype=tf.int64,
            value_index=tf.lookup.TextFileIndex.LINE_NUMBER
        )
        # Create a StaticVocabularyTable with OOV buckets
        self.table = tf.lookup.StaticVocabularyTable(
            self.init, num_oov, lookup_key_dtype=tf.string
        )

        # Add a dummy Dense layer to simulate a model's core network (from original example)
        self.dense = tf.keras.layers.Dense(1)
        self.dropout = tf.keras.layers.Dropout(rate=0.2)

    @tf.function
    def lookup(self, strings):
        # Lookup string tokens in the vocabulary table
        return self.table.lookup(strings)

    def call(self, inputs):
        # Inputs are assumed to be float tensors for the dense model branch
        # This call method just forwards input through the dense + dropout layers
        x = self.dense(inputs)
        x = self.dropout(x)
        return x

def my_model_function():
    # For demonstration, assume vocab_file is a fixed string path "test.file"
    # In real usage, user can change this path or pass as argument
    vocab_file = "./test.file"
    return MyModel(vocab_file=vocab_file)

def GetInput():
    # The model expects float inputs of shape (None, 1) for the dense branch,
    # but lookup table expects string inputs to lookup.
    # To directly test the lookup function, provide string input tensor,
    # but to test call(), provide float input to dense.

    # Here we produce example float input for the model's call method:
    float_input = tf.random.uniform((4, 1), dtype=tf.float32)  # Batch size 4, feature size 1

    # Also prepare string inputs for lookup testing (batch size 4 of strings):
    string_input = tf.constant([
        "here_is_one_row",
        "here_is_another_row",
        "some_unknown_token",
        "here_is_one_row"
    ])

    # For convenience, we return both inputs as a tuple, user can feed accordingly
    # If you want only single input matching call(), use float_input
    # If you want to test lookup(), pass string_input

    # Since the main call is on float input, return that for default usage
    return float_input

