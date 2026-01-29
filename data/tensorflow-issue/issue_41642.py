# tf.random.uniform((None), dtype=tf.string) ← inferred input shape: 1D string tensor with variable length sequence of tokens

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Scores and counts placeholders
        self.oov_score = 0
        self.pseudo_count = 5.0

        # These will be assigned after loading the data
        self.word_casing = None  # tf.lookup.StaticHashTable
        self.word_casing_lookup = None  # Dense tensor of alternative tokens per key

        self.tf_uni_dist = None  # tf.lookup.StaticHashTable
        self.backwardBiDist = None  # tf.lookup.StaticHashTable
        self.forwardBiDist = None  # tf.lookup.StaticHashTable
        self.trigramDist = None  # tf.lookup.StaticHashTable

    def fit_tf_lookup_table(self, dist):
        keys_tensor = tf.constant(list(dist.keys()))
        vals_tensor = tf.constant(list(dist.values()), dtype=tf.int64)
        initializer = tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_tensor)
        table = tf.lookup.StaticHashTable(initializer, self.oov_score)
        return table

    def fit_word_casing(self, wordCasingLookup):
        # wordCasingLookup: dict mapping word -> list of casing variants

        indices = []
        values = []
        tokens = []
        max_len = 0

        for idx, (token, items) in enumerate(wordCasingLookup.items()):
            tokens.append(token)
            max_len = max(max_len, len(items))
            for j, item in enumerate(items):
                indices.append([idx, j])
                values.append(item)

        word_casing_lookup_shape = [len(tokens), max_len]
        word_casing_lookup_tf = tf.SparseTensor(indices=indices, values=values, dense_shape=word_casing_lookup_shape)
        dense_word_casing_lookup_tf = tf.sparse.to_dense(word_casing_lookup_tf, default_value=b'')

        initializer = tf.lookup.KeyValueTensorInitializer(tf.constant(tokens), tf.range(len(tokens)))
        table = tf.lookup.StaticHashTable(initializer, -1)
        return table, dense_word_casing_lookup_tf

    def fit(self, wordCasingLookup, uniDist, backwardBiDist, forwardBiDist, trigramDist):
        self.word_casing, self.word_casing_lookup = self.fit_word_casing(wordCasingLookup)
        self.tf_uni_dist = self.fit_tf_lookup_table(uniDist)
        self.backwardBiDist = self.fit_tf_lookup_table(backwardBiDist)
        self.forwardBiDist = self.fit_tf_lookup_table(forwardBiDist)
        self.trigramDist = self.fit_tf_lookup_table(trigramDist)

    def get_alternative_tokens(self, possible_token_tensor):
        idx = self.word_casing.lookup(possible_token_tensor)

        def f1():
            return tf.constant([], dtype=tf.string)

        def f2():
            alternatives = self.word_casing_lookup[idx[0]]
            valid_alternatives = tf.gather(alternatives, tf.where(alternatives != b''))
            valid_alternatives = tf.reshape(valid_alternatives, [-1])
            return valid_alternatives

        alternative_tokens = tf.cond(tf.equal(idx, -1), f1, f2)
        return alternative_tokens

    def compute_unigram_score(self, possible_token_tensor, alternative_tokens):
        nominator = tf.cast(self.tf_uni_dist.lookup(possible_token_tensor), tf.float32) + self.pseudo_count
        denominator = self.tf_uni_dist.lookup(alternative_tokens)
        denominator_sum = tf.reduce_sum(tf.cast(denominator, tf.float32) + self.pseudo_count)
        score = nominator / denominator_sum
        return score

    def compute_bigram_backward_score(self, possible_token_tensor, prev_token_tensor, alternative_tokens):
        x = prev_token_tensor + tf.constant('_') + possible_token_tensor
        nominator = tf.cast(self.backwardBiDist.lookup(x), tf.float32) + self.pseudo_count
        alternative_tokens_m = prev_token_tensor + tf.constant('_') + alternative_tokens
        denominator = self.backwardBiDist.lookup(alternative_tokens_m)
        denominator_sum = tf.reduce_sum(tf.cast(denominator, tf.float32) + self.pseudo_count)
        score = nominator / denominator_sum
        return score

    def compute_bigram_forward_score(self, possible_token_tensor, next_token_tensor, alternative_tokens):
        x = possible_token_tensor + tf.constant('_') + tf.strings.lower(next_token_tensor)
        nominator = tf.cast(self.forwardBiDist.lookup(x), tf.float32) + self.pseudo_count
        alternative_tokens_m = alternative_tokens + tf.constant('_') + tf.strings.lower(next_token_tensor)
        denominator = self.forwardBiDist.lookup(alternative_tokens_m)
        denominator_sum = tf.reduce_sum(tf.cast(denominator, tf.float32) + self.pseudo_count)
        score = nominator / denominator_sum
        return score

    def compute_trigram_score(self, possible_token_tensor, prev_token_tensor, next_token_tensor, alternative_tokens):
        x = prev_token_tensor + tf.constant('_') + possible_token_tensor + tf.constant('_') + tf.strings.lower(next_token_tensor)
        nominator = tf.cast(self.trigramDist.lookup(x), tf.float32) + self.pseudo_count
        alternative_tokens_m = prev_token_tensor + tf.constant('_') + alternative_tokens + tf.constant('_') + tf.strings.lower(next_token_tensor)
        denominator = self.trigramDist.lookup(alternative_tokens_m)
        denominator_sum = tf.reduce_sum(tf.cast(denominator, tf.float32) + self.pseudo_count)
        score = nominator / denominator_sum
        return score

    def get_score(self, prev_token, possible_token, next_token):
        possible_token_l = tf.strings.lower(possible_token)
        alternative_tokens = self.get_alternative_tokens(possible_token_l)

        unigram_score = self.compute_unigram_score(possible_token, alternative_tokens)
        result = tf.math.log(unigram_score)

        if prev_token is not None:
            bigram_backward_score = self.compute_bigram_backward_score(possible_token, prev_token, alternative_tokens)
            result += tf.math.log(bigram_backward_score)

        if next_token is not None:
            bigram_forward_score = self.compute_bigram_forward_score(possible_token, next_token, alternative_tokens)
            result += tf.math.log(bigram_forward_score)

        if prev_token is not None and next_token is not None:
            trigram_score = self.compute_trigram_score(possible_token, prev_token, next_token, alternative_tokens)
            result += tf.math.log(trigram_score)
        return result

    def capitalize_str(self, token_tensor):
        token_tensor = tf.reshape(token_tensor, [1])
        char_tensor = tf.strings.unicode_split(token_tensor, 'UTF-8')
        first_char = char_tensor[0]
        first_char_cap = tf.strings.upper(first_char)
        cap_char_tensor = tf.concat([tf.reshape(first_char_cap, [1]), char_tensor[1:]], 0)
        cap_tensor = tf.strings.reduce_join(cap_char_tensor)
        return tf.reshape(cap_tensor, [1])

    @tf.function(input_signature=[tf.TensorSpec(shape=(None,), dtype=tf.string, name="input_text")])
    def call(self, tokens_tensor):
        # Aliased as the forward method. Use call to support tf.saved_model.save signature.
        return self.get_true_case(tokens_tensor)

    @tf.function(input_signature=[tf.TensorSpec(shape=(None,), dtype=tf.string)])
    def get_true_case(self, tokens_tensor):
        # Capitalize first token
        cap_first_token = self.capitalize_str(tokens_tensor[0:1])
        true_cased_tokens = cap_first_token

        # Tokens excluding first and last token for processing (note last token excluded as in original)
        tokens = tokens_tensor[1:-1]
        i = tf.constant(0, dtype=tf.int32)

        condition = lambda i, true_cased_tokens: tf.less(i, tf.size(tokens))

        def body(i, true_cased_tokens):
            cur_token = tokens[i:i+1]
            prev_token = true_cased_tokens[-1:]

            def f1():
                return tokens[i+1:i+2]

            def f2():
                return tf.constant([b''], dtype=tf.string)

            next_token = tf.cond(tf.less(i+1, tf.size(tokens)), f1, f2)

            word_casing_lookup = self.get_alternative_tokens(cur_token)

            def f_true_cased_0():
                return cur_token

            def f_true_cased_1():
                return word_casing_lookup[0:1]

            cur_token_transformed = tf.cond(tf.equal(tf.size(word_casing_lookup), 0),
                                            f_true_cased_0,
                                            f_true_cased_1)

            def f_return_cur():
                return cur_token_transformed

            def f_find_best():
                scores = tf.map_fn(lambda x: self.get_score(prev_token, x, next_token),
                                   word_casing_lookup, dtype=tf.float32)
                max_el_ind = tf.argmax(scores)
                truecased_token = word_casing_lookup[max_el_ind:max_el_ind+1]
                return truecased_token

            cur_token_transformed = tf.cond(tf.greater(tf.size(word_casing_lookup), 1), 
                                            f_find_best, 
                                            f_return_cur)

            cur_token_transformed = tf.reshape(cur_token_transformed, [-1])  # ensure rank-1 tensor

            true_cased_tokens = tf.concat([true_cased_tokens, cur_token_transformed], axis=0)

            return i + 1, true_cased_tokens

        _, res = tf.while_loop(condition,
                              body,
                              loop_vars=[i, true_cased_tokens],
                              shape_invariants=[tf.TensorShape([]), tf.TensorShape([None])])

        return res

def my_model_function():
    # Instantiate and return MyModel; weights need to be loaded externally via fit()
    return MyModel()

def GetInput():
    # Generate a 1D string tensor with some dummy tokens to test the model
    tokens = tf.constant([
        b'This', b'is', b'a', b'Sample', b'text', b'input', b'for', b'TrueCasing', b'Model', b'.'
    ])
    return tokens

