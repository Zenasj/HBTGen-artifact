import tensorflow as tf

class NGramTF(tf.Module):
    def __init__(self):
        self.oov_score = 0
        self.pseudo_count = 5.0
        pass

    def fit_tf_lookup_table(self, dist):
        keys_tensor = tf.constant(list(dist.keys()))
        vals_tensor = tf.constant(list(dist.values()))
        initializer = tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_tensor)
        table = tf.lookup.StaticHashTable(initializer, self.oov_score)
        return table

    def fit_word_casing(self, wordCasingLookup):
        indices = []
        values = []
        tokens = []
        for idx, (token, items) in enumerate(wordCasingLookup.items()):
            tokens.append(token)
            for j, item in enumerate(items):
                indices.append([idx, j])
                values.append(item)
        word_casing_lookup_shape = [len(tokens), max([len(item) for item in wordCasingLookup.values()])]
        word_casing_lookup_tf = tf.SparseTensor(indices=indices, values=values, dense_shape=word_casing_lookup_shape)
        word_casing_indices = tf.range(0, len(tokens))
        dense_word_casing_lookup_tf = tf.sparse.to_dense(word_casing_lookup_tf, default_value='')

        initializer = tf.lookup.KeyValueTensorInitializer(tf.constant(tokens), word_casing_indices)

        # TODO: this does ot work
        table = tf.lookup.StaticHashTable(initializer, -1)
        return table, dense_word_casing_lookup_tf

    def fit(self, wordCasingLookup, uniDist, backwardBiDist, forwardBiDist, trigramDist):
        self.word_casing, self.word_casing_lookup = self.fit_word_casing(wordCasingLookup)
        self.tf_uni_dist = self.fit_tf_lookup_table(uniDist)
        self.backwardBiDist = self.fit_tf_lookup_table(backwardBiDist)
        self.forwardBiDist = self.fit_tf_lookup_table(forwardBiDist)
        self.trigramDist = self.fit_tf_lookup_table(trigramDist)

def load_truecasing_model(model_filename):
    with open(model_filename, 'rb') as bin_file:  # from s3://workfit-models/auto-punc/
        uni_dist = pickle.load(bin_file)
        backward_bi_dist = pickle.load(bin_file)
        forward_bi_dist = pickle.load(bin_file)
        trigram_dist = pickle.load(bin_file)
        word_casing_lookup = pickle.load(bin_file)
        return word_casing_lookup, uni_dist, backward_bi_dist, forward_bi_dist, trigram_dist

truecaser_weights = 'en/en_truecasing_model.obj'
export_path = './truecaser_serving/1/'
wordCasingLookup, uniDist, backwardBiDist, forwardBiDist, trigramDist = load_truecasing_model(truecaser_weights)
tf_model = truecaser_tf.NGramTF()
tf_model.fit(wordCasingLookup, uniDist, backwardBiDist, forwardBiDist, trigramDist)
signature_def = tf_model.get_true_case.get_concrete_function(
        tf.TensorSpec(shape=(None), dtype=tf.string, name="input_text"))
tf.saved_model.save(tf_model,export_path,signatures={'serving_default': signature_def})