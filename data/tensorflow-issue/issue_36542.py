import tensorflow as tf

class RNNBatch:
    def __init__(self, rnn_layer, rnn_batch_size, true_words, true_lengths, training):
        self.rnn_batch_size = rnn_batch_size
        self.training = training
        self.dtype = tf.float32

        self.rnn_layer = rnn_layer
        max_sequence_len = rnn_layer.max_sequence_len
        dictionary_size = rnn_layer.dictionary_size

        self.true_words = true_words
        self.true_lengths = true_lengths

        self.written = 0
        self.rnn_processed_start = 0

        self.output_written = 0

    def run(self, arrays):
        selected_features = arrays['selected_features'].concat()

        batch_size = tf.shape(selected_features)[0]
        states_h = tf.zeros((batch_size, self.rnn_layer.num_rnn_units), dtype=self.dtype)
        states_c = tf.zeros((batch_size, self.rnn_layer.num_rnn_units), dtype=self.dtype)
        states = [states_h, states_c]

        tw = self.true_words[self.rnn_processed_start : self.rnn_processed_start + self.written, ...]
        tl = self.true_lengths[self.rnn_processed_start : self.rnn_processed_start + self.written, ...]

        out, out_ar = self.rnn_layer(selected_features, tw, tl, states, self.training)

        arrays['outputs'] = arrays['outputs'].write(self.output_written, out)
        arrays['outputs_ar'] = arrays['outputs'].write(self.output_written, out_ar)
        self.output_written += 1

        self.rnn_processed_start += self.written
        self.written = 0

        return arrays

    def feed_crop(self, cropped_features, arrays):
        arrays['selected_features'] = arrays['selected_features'].write(self.written, cropped_features)
        self.written += 1

        if self.written == self.rnn_batch_size:
            arrays = self.run(arrays)
            arrays['selected_features'] = tf.TensorArray(cropped_features.dtype, size=0, element_shape=tf.TensorShape([None] + [cropped_features.shape[1:]]))


        return arrays

    def return_values(self, arrays):
        if self.written != 0:
            arrays = self.run(arrays)

        return arrays['outputs'].concat(), arrays['outputs_ar'].concat()