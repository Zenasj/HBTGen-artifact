# tf.random.uniform((None,), dtype=tf.string) ← The model takes a 1D string tensor input of variable batch size

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, vocab_file: str):
        super().__init__()
        # To fix the asset tracking issue described in the GitHub issue,
        # keep the initializer as an explicit attribute so its assets are tracked.
        self._vocab_initializer = tf.lookup.TextFileInitializer(
            vocab_file,
            key_dtype=tf.string,
            key_index=tf.lookup.TextFileIndex.WHOLE_LINE,
            value_dtype=tf.int64,
            value_index=tf.lookup.TextFileIndex.LINE_NUMBER)
        # StaticHashTable created with the initializer
        self._vocab_table = tf.lookup.StaticHashTable(
            self._vocab_initializer,
            default_value=-1)

    @tf.function(input_signature=[tf.TensorSpec(shape=(None,), dtype=tf.string)])
    def call(self, inputs):
        # Lookup input strings in vocab table
        return self._vocab_table.lookup(inputs)

def my_model_function():
    # Provide a dummy vocab file to the model as required.
    # For demonstration, create an in-memory tempfile with vocab contents.
    import tempfile
    import os

    vocab_lines = ["a", "b", "c", "d"]
    # Use a temporary directory for vocab file asset.
    temp_dir = tempfile.mkdtemp()
    vocab_path = os.path.join(temp_dir, "vocab.txt")

    with open(vocab_path, "w") as f:
        for line in vocab_lines:
            f.write(line + "\n")

    # Return instance of MyModel with vocab file path
    return MyModel(vocab_path)

def GetInput():
    # Return a batch of sample input strings matching expected input signature
    # The input shape is (batch_size,), dtype string
    sample_inputs = tf.constant(["a", "b", "x", "d"], dtype=tf.string)
    return sample_inputs

