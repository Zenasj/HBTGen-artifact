# tf.random.uniform((B, T), dtype=tf.int32) ← Input is (batch_size, sequence_length) integer token IDs

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
import numpy as np

# This code is adapted from a custom CRF layer example on TF 1.x,
# modernized to TensorFlow 2.x style with tf.function compatibility assumptions.
# Original issue involved model crashing on fit() with large batch size and validation_split=0.0,
# caused by dimension mismatch in metric computation.

# We assume input shape is (batch_size, sequence_length) with integer token IDs,
# which are embedded, passed through a TimeDistributed Dense layer to produce
# unary potentials for CRF tag inference.


class CRF(keras.layers.Layer):
    def __init__(self, num_tags, **kwargs):
        super(CRF, self).__init__(**kwargs)
        self.num_tags = num_tags
        self.input_spec = keras.layers.InputSpec(min_ndim=3)
        self.supports_masking = True

    def get_config(self):
        config = {'num_tags': self.num_tags}
        base_config = super(CRF, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        # input_shape: (batch_size, sequence_length, num_tags)
        if len(input_shape) != 3:
            raise ValueError('CRF expects input shape (batch, seq_len, num_tags)')
        if input_shape[-1] is None:
            raise ValueError('The last dimension of the inputs to `CRF` should be defined.')
        if input_shape[-1] != self.num_tags:
            raise ValueError(
                'The last dimension of the input shape must be equal to num_tags.')
        self.transitions = self.add_weight(
            name='transitions',
            shape=(self.num_tags, self.num_tags),
            initializer='glorot_uniform',
            trainable=True)
        super(CRF, self).build(input_shape)

    def call(self, inputs, mask=None):
        # inputs are unary potentials (logits) for each tag at each timestep
        seq_lens = get_seq_lens(inputs, mask)
        # Use tf_adaptive or tf-compat viterbi decode
        # TF 2.x migration: tf.contrib.crf is deprecated,
        # Use tf_addons if available, else mimic
        # For compatibility, we use tf.raw_ops or fallback

        # For demonstration, we use tf_addons (must be installed)
        # Here, we do a fallback:
        try:
            import tensorflow_addons as tfa
            viterbi_sequence, _ = tfa.text.crf_decode(
                inputs, self.transitions, seq_lens)
        except ImportError:
            # Fallback: dummy argmax decoding - not real Viterbi, for demo only
            viterbi_sequence = tf.argmax(inputs, axis=-1, output_type=tf.int32)

        outputs = K.one_hot(viterbi_sequence, self.num_tags)
        # During training, output unary potentials (for loss computation)
        # During inference (test phase), output decoded sequence one-hot
        return K.in_train_phase(inputs, outputs)

    def compute_output_shape(self, input_shape):
        # output shape same as input (batch, seq_len, num_tags)
        return input_shape

    def compute_mask(self, inputs, mask=None):
        if mask is not None:
            # Retain mask for sequence dimension
            return mask
        return mask


def get_seq_lens(inputs, mask=None):
    # inputs: (batch, seq_len, num_tags)
    if mask is not None:
        return K.sum(K.cast(mask, dtype='int32'), axis=-1)
    else:
        shape = K.shape(inputs)
        return tf.fill([shape[0]], shape[1])


def crf_loss(y_true, y_pred):
    # y_true, y_pred shape: (batch, seq_len, num_tags)
    # Extract CRF layer and input info
    crf, idx = y_pred._keras_history[:2]
    inputs = crf.get_input_at(idx)
    mask = crf.get_input_mask_at(idx)
    seq_lens = get_seq_lens(inputs, mask)

    # Convert one-hot to label indices
    y_true_labels = K.cast(K.argmax(y_true, axis=-1), dtype='int32')

    try:
        import tensorflow_addons as tfa
        log_likelihood, _ = tfa.text.crf_log_likelihood(
            inputs, y_true_labels, seq_lens, transition_params=crf.transitions)
    except ImportError:
        # Backup dummy loss using categorical crossentropy (valid only if no mask)
        # This is a fallback and does not reflect real CRF loss
        return K.mean(K.categorical_crossentropy(y_true, y_pred, from_logits=True))

    return K.mean(-log_likelihood)


def crf_accuracy(y_true, y_pred):
    # y_true, y_pred shape: (batch, seq_len, num_tags)
    crf, idx = y_pred._keras_history[:2]
    inputs = crf.get_input_at(idx)
    mask = crf.get_input_mask_at(idx)
    seq_lens = get_seq_lens(inputs, mask)

    try:
        import tensorflow_addons as tfa
        viterbi_sequence, _ = tfa.text.crf_decode(inputs, crf.transitions, seq_lens)
    except ImportError:
        viterbi_sequence = tf.argmax(inputs, axis=-1, output_type=tf.int32)

    y_true_labels = K.cast(K.argmax(y_true, -1), dtype='int32')
    correct = K.cast(K.equal(viterbi_sequence, y_true_labels), K.floatx())

    if mask is None:
        return K.mean(correct)
    else:
        mask = K.cast(mask, K.floatx())
        return K.sum(correct * mask) / K.sum(mask)


num_words = 10  # vocabulary size for embedding input IDs
num_features = 100  # embedding dimension
num_tags = 5   # number of CRF tags


class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.embedding = keras.layers.Embedding(
            input_dim=num_words, output_dim=num_features, mask_zero=True)
        self.dense = keras.layers.TimeDistributed(
            keras.layers.Dense(num_tags))
        self.crf = CRF(num_tags)

    def call(self, inputs, training=None, mask=None):
        x = self.embedding(inputs)
        x = self.dense(x)
        x = self.crf(x)
        return x


def my_model_function():
    model = MyModel()
    # Build the model by providing example input shape
    model.build(input_shape=(None, None))  # (batch, seq_len)
    # Compile with optimizer, loss and metrics as defined
    model.compile(
        optimizer='adam',
        loss=crf_loss,
        metrics=[crf_accuracy]
    )
    return model


def GetInput():
    # Return a random input tensor compatible with MyModel:
    # Shape: (batch_size=2, seq_len=6), integer token IDs from 1 to num_words-1,
    # padding with 0 to simulate masked tokens.
    x = np.array([[1, 2, 3, 4, 0, 0],
                  [4, 5, 6, 0, 0, 0]])
    return tf.convert_to_tensor(x, dtype=tf.int32)

