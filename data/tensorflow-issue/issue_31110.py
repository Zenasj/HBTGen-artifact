from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
# import keras
# from keras import backend as K
import numpy as np


class CRF(keras.layers.Layer):

    def __init__(self, num_tags, **kwargs):
        super(CRF, self).__init__(**kwargs)
        self.num_tags = num_tags
        self.input_spec = keras.layers.InputSpec(min_ndim=3)
        self.supports_masking = True

    def get_config(self):
        config = {
            'num_tags': self.num_tags,
        }
        base_config = super(CRF, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        assert len(input_shape) == 3
        if input_shape[-1] is None:
            raise ValueError('The last dimension of the inputs to `CRF` '
                             'should be defined. Found `None`.')
        if input_shape[-1] != self.num_tags:
            raise ValueError('The last dimension of the input shape must be equal to output'
                             ' shape. Use a linear layer if needed.')
        self.transitions = self.add_weight(name='transitions',
                                           shape=[self.num_tags,
                                                  self.num_tags],
                                           initializer="glorot_uniform",
                                           trainable=True)
        self.built = True

    def call(self, inputs, mask=None):
        seq_lens = get_seq_lens(inputs, mask)
        viterbi_sequence, _ = tf.contrib.crf.crf_decode(inputs,
                                                        self.transitions,
                                                        seq_lens)
        outputs = K.one_hot(viterbi_sequence, self.num_tags)
        return K.in_train_phase(inputs, outputs)

    def compute_output_shape(self, input_shape):
        return input_shape[:2] + (self.num_tags,)

    def compute_mask(self, inputs, mask=None):
        if mask is not None:
            return K.any(mask, axis=1)
        return mask


def get_seq_lens(inputs, mask=None):
    if mask is not None:
        return K.sum(K.cast(mask, dtype='int32'), axis=-1)
    else:
        shape = K.int_shape(inputs)
        return K.ones(shape[:-1], dtype='int32') * shape[-1]


def crf_loss(y_true, y_pred):
    crf, idx = y_pred._keras_history[:2]
    inputs = crf.get_input_at(idx)
    mask = crf.get_input_mask_at(idx)
    seq_lens = get_seq_lens(inputs, mask)
    y_true = K.cast(K.argmax(y_true, axis=-1), dtype='int32')
    log_likelihood, crf.transitions = \
        tf.contrib.crf.crf_log_likelihood(y_pred,
                                          y_true,
                                          seq_lens,
                                          transition_params=crf.transitions)
    return K.mean(-log_likelihood)


def crf_accuracy(y_true, y_pred):
    crf, idx = y_pred._keras_history[:2]
    inputs = crf.get_input_at(idx)
    mask = crf.get_input_mask_at(idx)
    seq_lens = get_seq_lens(inputs, mask)
    viterbi_sequence, _ = tf.contrib.crf.crf_decode(inputs,
                                                    crf.transitions,
                                                    seq_lens)
    y_true = K.cast(K.argmax(y_true, -1), dtype='int32')
    judge = K.cast(K.equal(viterbi_sequence, y_true), K.floatx())
    if mask is None:
        return K.mean(judge)
    else:
        mask = K.cast(mask, K.floatx())
        return K.sum(judge * mask) / K.sum(mask)


num_words = 20
num_features = 100
num_tags = 5

inputs = keras.layers.Input(shape=(None,))
embedding = keras.layers.Embedding(10, num_features, mask_zero=True)(inputs)
scores = keras.layers.TimeDistributed(keras.layers.Dense(num_tags))(embedding)
crf = CRF(num_tags)
outputs = crf(scores)
model = keras.models.Model(inputs, outputs)

model.summary()

x = np.array([[1, 2, 3, 4, 0, 0], [4, 5, 6, 0, 0, 0]])
y = np.array([[1, 3, 4, 2, 0, 0], [2, 1, 3, 0, 0, 0]])
y = np.eye(num_tags)[y]

print(x)
print(x.shape)
print(y)
print(y.shape)

model.compile(optimizer="adam",
              loss=crf_loss,
              metrics=[crf_accuracy])

model.fit(x, y, batch_size=16, epochs=50, validation_split=0.0)