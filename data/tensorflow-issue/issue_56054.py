# tf.repeat(tf.constant(['A', 'B', 'C']), 10) (shape=(30,), dtype=string)
import tensorflow as tf

# This is a fused and adapted model combining the original TestModel and the
# proposed patch that modifies the internal lookup table to be mutable, enabling
# proper saving and loading of weights (vocabulary) in StringLookup layers.

class MyModel(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Patch StringLookup to use MutableHashTable internally so the vocabulary can be saved and restored
        self._patch_string_lookup()
        self.sl = tf.keras.layers.StringLookup()

    def _patch_string_lookup(self):
        StringLookup = tf.keras.layers.StringLookup

        def _uninitialized_lookup_table(self):
            # Use a MutableHashTable instead of StaticHashTable for checkpointing support
            with tf.init_scope():
                return tf.lookup.experimental.MutableHashTable(
                    key_dtype=self._key_dtype,
                    value_dtype=self._value_dtype,
                    default_value=self._default_value,
                    checkpoint=True  # enable checkpointing for serialization
                )

        def _lookup_table_from_tokens(self, tokens):
            with tf.init_scope():
                lookup_table = self._uninitialized_lookup_table()
                token_start = self._token_start_index()
                token_end = token_start + tf.size(tokens)
                indices = tf.range(token_start, token_end, dtype=tf.int64)
                keys, values = (indices, tokens) if self.invert else (tokens, indices)
                lookup_table.insert(keys, values)
                return lookup_table

        # Patch the StringLookup methods - monkey patching class methods
        StringLookup._uninitialized_lookup_table = _uninitialized_lookup_table
        StringLookup._lookup_table_from_tokens = _lookup_table_from_tokens

    def adapt(self, dataset):
        # Adapt vocabulary from a tf.data.Dataset or tensor
        self.sl.adapt(dataset)
    
    def call(self, x):
        # Forward pass: converts string input to integer indices using adapted vocabulary
        return self.sl(x)


def my_model_function():
    # Instantiate and return the patched MyModel
    model = MyModel()
    return model


def GetInput():
    # Generate input tensor matching the expected input of MyModel:
    # A 1D tensor of strings, containing strings from vocab ['A','B','C'] repeated.
    inp = tf.repeat(tf.constant(['A', 'B', 'C']), 10)
    return inp

