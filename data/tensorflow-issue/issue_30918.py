from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import backend as K
import sys

from BertLayer import BertLayer
from preprocessing import MyDocs

sess = tf.Session()

def build_model(bert_path, max_seq_length):
    in_id = tf.keras.layers.Input(shape=(max_seq_length,), name="input_ids")
    in_mask = tf.keras.layers.Input(shape=(max_seq_length,), name="input_masks")
    in_segment = tf.keras.layers.Input(shape=(max_seq_length,), name="segment_ids")
    bert_inputs = [in_id, in_mask, in_segment]

    bert_output = BertLayer(n_fine_tune_layers=3, bert_path=bert_path, pooling="first")(bert_inputs)
    dense = tf.keras.layers.Dense(256, activation="relu")(bert_output)
    pred = tf.keras.layers.Dense(5, activation="sigmoid")(dense)  # change this to build classifier

    model = tf.keras.models.Model(inputs=bert_inputs, outputs=pred)
    model.compile(loss="binary_crossentropy", optimizer="adam",  metrics=['binary_accuracy', 'categorical_accuracy'])
    model.summary()

    return model


def initialize_vars(sess):
    sess.run(tf.compat.v1.local_variables_initializer())
    sess.run(tf.compat.v1.global_variables_initializer())
    sess.run(tf.compat.v1.tables_initializer())
    K.set_session(sess)

def main():
    # Params for bert model and tokenization
    bert_path = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"
    max_seq_length = 256

    corpus = MyDocs("datasets/bbc/raw", bert_path, max_seq_length)

    ids = []
    masks = []
    segment_ids = []
    for id, mask, segment, label in corpus:
        ids.append(id)
        masks.append(masks)
        segment_ids.append(segment)
    X = [ids, masks, segment_ids]

    labels = corpus.labels
    label_encoder = OneHotEncoder()
    y = label_encoder.fit_transform(np.array(labels).reshape(-1, 1)).todense()
    print('Dimension of labels input is {}.'.format(y.shape))

    print('Building model...')
    model = build_model(bert_path, max_seq_length)

    print('Training model...')
    history = model.fit(X, y,
                        validation_split=0.2,
                        epochs=1,
                        batch_size=1,
                        verbose=2,
                        use_multiprocessing=True)

if __name__ == "__main__":
    main()

import tensorflow as tf
from tensorflow.keras import backend as K
import tensorflow_hub as hub

class BertLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        n_fine_tune_layers=10,
        pooling="mean",
        bert_path="https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1",
        **kwargs,
    ):
        self.n_fine_tune_layers = n_fine_tune_layers
        self.trainable = True
        self.output_size = 768
        self.pooling = pooling
        self.bert_path = bert_path

        if self.pooling not in ["first", "mean"]:
            raise NameError(
                f"Undefined pooling type (must be either first or mean, but is {self.pooling}"
                )

        super(BertLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.bert = hub.Module(
            self.bert_path, trainable=self.trainable, name=f"{self.name}_module"
        )

        # Remove unused layers
        trainable_vars = self.bert.variables
        if self.pooling == "first":
            trainable_vars = [var for var in trainable_vars if not "/cls/" in var.name]
            trainable_layers = ["pooler/dense"]

        elif self.pooling == "mean":
            trainable_vars = [
                var
                for var in trainable_vars
                if not "/cls/" in var.name and not "/pooler/" in var.name
            ]
            trainable_layers = []
        else:
            raise NameError(
                f"Undefined pooling type (must be either first or mean, but is {self.pooling}"
            )

        # Select how many layers to fine tune
        for i in range(self.n_fine_tune_layers):
            trainable_layers.append(f"encoder/layer_{str(11 - i)}")

        # Update trainable vars to contain only the specified layers
        trainable_vars = [
            var
            for var in trainable_vars
            if any([l in var.name for l in trainable_layers])
        ]

        # Add to trainable weights
        for var in trainable_vars:
            self._trainable_weights.append(var)

        for var in self.bert.variables:
            if var not in self._trainable_weights:
                self._non_trainable_weights.append(var)

        super(BertLayer, self).build(input_shape)

    def call(self, inputs):
        inputs = [K.cast(x, dtype="int32") for x in inputs]
        input_ids, input_mask, segment_ids = inputs
        bert_inputs = dict(
            input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids
        )
        if self.pooling == "first":
            # pooled output of the entire sequence [batch, hidden_size]
            pooled = self.bert(inputs=bert_inputs, signature="tokens", as_dict=True)["pooled_output"]
        elif self.pooling == "mean":
            # representation of every token in the sequence [batch, max_seq_length, hidden_size]
            result = self.bert(inputs=bert_inputs, signature="tokens", as_dict=True)["sequence_output"]

            mul_mask = lambda x, m: x * tf.expand_dims(m, axis=-1)
            masked_reduce_mean = lambda x, m: tf.reduce_sum(mul_mask(x, m), axis=1) / (
                    tf.reduce_sum(m, axis=1, keepdims=True) + 1e-10)
            input_mask = tf.cast(input_mask, tf.float32)
            pooled = masked_reduce_mean(result, input_mask)
        else:
            raise NameError(f"Undefined pooling type (must be either first or mean, but is {self.pooling}")

        return pooled

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_size)

def build_model(bert_path, max_seq_length, tune_cells):
    in_id = tf.keras.layers.Input(shape=(max_seq_length,), name="input_ids")
    in_mask = tf.keras.layers.Input(shape=(max_seq_length,), name="input_masks")
    in_segment = tf.keras.layers.Input(shape=(max_seq_length,), name="segment_ids")
    bert_inputs = [in_id, in_mask, in_segment]

    bert_output = BERT(finetune_cells=tune_cells, bert_path=bert_path)(bert_inputs)
    pooled = tf.keras.layers.GlobalMaxPooling1D()(bert_output)
    dense = tf.keras.layers.Dense(256, activation="relu")(pooled)
    pred = tf.keras.layers.Dense(5, activation="sigmoid")(dense)

    model = tf.keras.models.Model(inputs=bert_inputs, outputs=pred)
    model.compile(loss="binary_crossentropy", optimizer="adam",  metrics=['binary_accuracy', 'categorical_accuracy'])
    model.summary()

    return model

def set_session(USE_XLA=True, MIXED_PRECISION=True):
    """
    Set the session config to optimise GPU computations.
    - Automatic Mixed Precision is available for server grade GPUs.
    """
    config = tf.ConfigProto()
    if USE_XLA:
        opt_level = tf.OptimizerOptions.ON_1
        tf.enable_resource_variables()
    else:
        opt_level = tf.OptimizerOptions.OFF

    config.graph_options.optimizer_options.global_jit_level = opt_level

    config.graph_options.rewrite_options.auto_mixed_precision = MIXED_PRECISION

    sess = tf.Session(config=config)
    tf.keras.backend.set_session(sess)

    return tf.keras.backend.get_session()

def initialize_sess(sess):
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
    tf.keras.backend.set_session(sess)

    K.set_session(sess)

def main():
    """
    0. Set Global Variables
    """
    sess = set_session()
    bert_path = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"
    max_seq_length = 256
    tune_layers = 1
    epochs = 2
    batch_size = 64

    """
    1. Build Data Set
    """
    corpus = MyDocs("datasets/bbc/raw", sess, bert_path, max_seq_length)

    ids = []
    masks = []
    segment_ids = []
    for id, mask, segment, label in corpus:
        ids.append(id)
        masks.append(masks)
        segment_ids.append(segment)
    X = [ids, masks, segment_ids]

    labels = corpus.labels
    label_encoder = OneHotEncoder()
    y = label_encoder.fit_transform(np.array(labels).reshape(-1, 1)).todense()
    print('Dimension of labels input is {}.'.format(y.shape))

    """
    2. Build Model and Train
    """
    print('Building model...')
    model = build_model(bert_path, max_seq_length, tune_layers)

    # Instantiate variables
    initialize_sess(sess)

    print('Training model...')
    history = model.fit(X, y,
                        validation_split=0.2,
                        epochs=epochs,
                        batch_size=batch_size,
                        verbose=2,
                        use_multiprocessing=True)