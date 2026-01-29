# tf.random.uniform((B,), dtype=tf.string) ← Input is a 1D tensor of strings of unknown batch size

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.oov_score = 0
        self.pseudo_count = 5.0
        # Initialize lookup tables as None; they will be set in fit()
        self.word_casing = None  # StaticHashTable for word casing lookup
        self.word_casing_lookup = None  # Dense tensor lookup of casing variants
        self.tf_uni_dist = None  # unigram distribution lookup table
        self.backwardBiDist = None
        self.forwardBiDist = None
        self.trigramDist = None

    def fit_tf_lookup_table(self, dist):
        # Helper to create a StaticHashTable from a string->float dictionary
        keys_tensor = tf.constant(list(dist.keys()))
        vals_tensor = tf.constant(list(dist.values()))
        initializer = tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_tensor)
        table = tf.lookup.StaticHashTable(initializer, default_value=self.oov_score)
        return table

    def fit_word_casing(self, wordCasingLookup):
        # wordCasingLookup: dict[token: list of strings]
        indices = []
        values = []
        tokens = []
        for idx, (token, items) in enumerate(wordCasingLookup.items()):
            tokens.append(token)
            for j, item in enumerate(items):
                indices.append([idx, j])
                values.append(item)
        max_len = max(len(item) for item in wordCasingLookup.values()) if wordCasingLookup else 0
        word_casing_lookup_shape = [len(tokens), max_len]
        # SparseTensor to dense tensor representation of casing variants
        word_casing_lookup_tf = tf.sparse.to_dense(
            tf.SparseTensor(indices=indices,
                            values=values,
                            dense_shape=word_casing_lookup_shape),
            default_value='')
        word_casing_indices = tf.range(0, len(tokens))
        initializer = tf.lookup.KeyValueTensorInitializer(
            tf.constant(tokens), word_casing_indices, key_dtype=tf.string, value_dtype=tf.int32)

        table = tf.lookup.StaticHashTable(initializer, default_value=-1)
        return table, word_casing_lookup_tf

    def fit(self, wordCasingLookup, uniDist, backwardBiDist, forwardBiDist, trigramDist):
        # Set all lookup tables and related variables
        self.word_casing, self.word_casing_lookup = self.fit_word_casing(wordCasingLookup)
        self.tf_uni_dist = self.fit_tf_lookup_table(uniDist)
        self.backwardBiDist = self.fit_tf_lookup_table(backwardBiDist)
        self.forwardBiDist = self.fit_tf_lookup_table(forwardBiDist)
        self.trigramDist = self.fit_tf_lookup_table(trigramDist)

    @tf.function(input_signature=[tf.TensorSpec(shape=(None,), dtype=tf.string)])
    def get_true_case(self, input_text):
        # Simplified truecaser logic:
        # 1) Lookup each token in word_casing table to get the index row for the casing variants.
        # 2) If token not found, return token unchanged.
        # 3) If found, pick the most frequent casing variant from word_casing_lookup based on unigram dist.
        # For this example, just return token itself if no info, or the first variant available.

        def truecase_token(token):
            idx = self.word_casing.lookup(token)
            # Return lowercased token if not found
            if tf.equal(idx, -1):
                return token
            # Otherwise get casing variants for this token row
            variants = self.word_casing_lookup[idx]
            # Filter out empty strings
            variants_non_empty = tf.boolean_mask(variants, tf.not_equal(variants, ''))
            if tf.shape(variants_non_empty)[0] == 0:
                return token
            # Return first variant as a placeholder for "most likely" casing
            return variants_non_empty[0]

        # Vectorize truecase_token over input_text
        output = tf.map_fn(truecase_token, input_text, dtype=tf.string)
        return output

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random tensor input consistent with the model's expected input (1D string tensor)
    # Generate random dummy tokens from ASCII lowercase letters for example
    batch_size = 8  # arbitrary
    import string
    import random
    # Generate dummy tokens ("word0", "word1", ..) for simplicity
    tokens = [f"word{i}" for i in range(batch_size)]
    return tf.constant(tokens, dtype=tf.string)

