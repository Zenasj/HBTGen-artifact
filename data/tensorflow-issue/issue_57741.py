from tensorflow import keras
from tensorflow.keras import layers

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""demo of preprcoessing large layers
"""

# pylint:disable=no-member
import time
from typing import Dict

import tensorflow as tf


@tf.keras.utils.register_keras_serializable()
def split_vertical(x):
    """
    用于TextVectorization的split入参, split by `|`
    """
    return tf.strings.split(x, sep="|")


# 25242127 -> 3323 seconds
class TestPreprocessor(tf.keras.Model):
    def __init__(self, name="test_preprocessor", vocab_size: int = 100, **kwargs):
        super().__init__(name=name, trainable=False, **kwargs)
        self._lookup = {}
        self.vocab_size = vocab_size
        print(f"\nvocab_size={vocab_size}")

    def adapt(self):
        vocab = [str(i) for i in range(self.vocab_size)]

        tic = time.time()
        text_vectorizer = tf.keras.layers.TextVectorization(
            vocabulary=vocab,
            output_mode="int",
            split=split_vertical,
            standardize=None,
            output_sequence_length=7,
            name="text_vectorization_" + "faked_id",
        )
        toc = time.time()
        print(f"Construct TextVectorization: {toc - tic:.2f} seconds")
        self._lookup["faked_id"] = text_vectorizer

    def save(self, filepath: str = "/tmp/debug_large_vocab"):
        ds = tf.data.Dataset.from_tensor_slices({"faked_id": [str(e) for e in range(100)]}).batch(3).map(self)
        _ = next(iter(ds))
        tic = time.time()
        super().save(filepath)
        toc = time.time()
        print(f"Save TextVectorzation:       {toc-tic:.2f} seconds")

    def call(self, batch: Dict):
        ans = {}
        for f, lookup in self._lookup.items():
            ans[f] = lookup(batch[f])
        return ans


def main():
    vsize = [1000, 10000, 100000, 1000000, 25000000]
    for V in vsize:
        preprocessor = TestPreprocessor(vocab_size=V)
        preprocessor.adapt()
        preprocessor.save()


if __name__ == "__main__":
    main()

python
text_vectorizer._self_tracked_trackables.pop()
text_vectorizer._lookup_layer._self_tracked_trackables.pop()