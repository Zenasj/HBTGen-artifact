# tf.random.uniform((B, ), dtype=tf.string) ← Input is a dict of string tensors keyed by "faked_id", batch size unknown B

import tensorflow as tf
import time
from typing import Dict

@tf.keras.utils.register_keras_serializable()
def split_vertical(x):
    """
    Used as split arg for TextVectorization: splits input strings by '|'
    """
    return tf.strings.split(x, sep="|")

class MyModel(tf.keras.Model):
    def __init__(self, name="test_preprocessor", vocab_size: int = 100, **kwargs):
        super().__init__(name=name, trainable=False, **kwargs)
        self._lookup = {}
        self.vocab_size = vocab_size
        # Will hold the TextVectorization layer

    def adapt(self):
        # Construct vocabulary as string representation of range(vocab_size)
        vocab = [str(i) for i in range(self.vocab_size)]

        # Create TextVectorization layer with given vocab
        # Use output_mode=int, no standardization, split by '|', fixed output length 7
        tic = time.time()
        text_vectorizer = tf.keras.layers.TextVectorization(
            vocabulary=vocab,
            output_mode="int",
            split=split_vertical,
            standardize=None,
            output_sequence_length=7,
            name="text_vectorization_faked_id",
        )
        toc = time.time()
        print(f"Construct TextVectorization: {toc - tic:.2f} seconds")

        # Fix for performance / save slowness:
        # Remove large tracked lists that cause slow saving (from issue comments)
        if hasattr(text_vectorizer, "_self_tracked_trackables"):
            if text_vectorizer._self_tracked_trackables:
                # Remove last item if suspected large list causing slow save
                text_vectorizer._self_tracked_trackables.pop()
        if hasattr(text_vectorizer._lookup_layer, "_self_tracked_trackables"):
            if text_vectorizer._lookup_layer._self_tracked_trackables:
                text_vectorizer._lookup_layer._self_tracked_trackables.pop()

        self._lookup["faked_id"] = text_vectorizer

    def call(self, batch: Dict[str, tf.Tensor]):
        # batch is expected to be dict {"faked_id": tf.Tensor of strings}
        ans = {}
        for f, lookup in self._lookup.items():
            ans[f] = lookup(batch[f])
        return ans

def my_model_function():
    # Initialize MyModel with a default reasonable vocab size
    model = MyModel(vocab_size=1000)
    model.adapt()
    return model

def GetInput():
    # Create a dict with key "faked_id" mapped to a batch of string inputs
    # Input strings are numeric strings in range vocab_size to match vocabulary for TextVectorization

    batch_size = 3  # example batch size
    # Random selection of strings "0" to "999"
    import numpy as np
    indices = np.random.randint(0, 1000, size=batch_size)
    strings = [str(i) for i in indices]
    # We use a tf.constant string tensor shaped [batch_size]
    input_dict = {"faked_id": tf.constant(strings, dtype=tf.string)}
    return input_dict

