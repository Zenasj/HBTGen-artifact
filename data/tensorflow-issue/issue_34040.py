from tensorflow import keras
from tensorflow.keras import layers

from typing import Any, Dict

import tensorflow as tf


from typing import Text, Tuple, List, Dict, Any, Union

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.python.ops import math_ops


def _split_iob_tag(tag: Text) -> Tuple[Text, Union[Text, None]]:
    """Handle tags <pad>, <unk>, O, B-foo, and I-foo."""
    if tag[:2] not in ["B-", "I-"]:
        return (tag, None)
    return tuple(tag.split("-", maxsplit=2))  # type: ignore


def iob_transition_params(tags: List[Text]) -> tf.constant:
    """Return transition matrix suitable for tf.contrib.crf.crf_decode."""
    mat = np.zeros((len(tags), len(tags)))
    for i, i_tag in enumerate(tags):
        i_type, i_name = _split_iob_tag(i_tag)
        for j, j_tag in enumerate(tags):
            j_type, j_name = _split_iob_tag(j_tag)
            if (i_type == "O" and j_type == "I") or (
                i_type in ["B", "I"] and j_type == "I" and i_name != j_name
            ):
                mat[i][j] = -np.inf
    return tf.constant(mat, dtype=tf.float32)


if __name__ == "__main__":
    params: Dict[str, Any] = {
        "dim": 300,
        "dropout": 0.5,
        "batch_size": 3,
        "buffer": 15000,
        "lstm_size": 10,
        "eval_dir": "../results/",
        "saved_model_dir": "../final_model/",
        "training_data": "some_data",
        "window_length": 512,
        "embedding_dimension": 300,
    }

    tags = {'I-tag1', 'B-tag1','I-tag2', 'B-tag2','I-tag3', 'B-tag3', 'O'}

    dropout = params["dropout"]

    embeddings = tf.keras.Input(
        shape=(params["window_length"], params["embedding_dimension"],),
        dtype=tf.float32,
        name="embedding_sequence",
    )
    nwords = tf.keras.Input(shape=(), dtype=tf.int32, name="nwords",)
    lstm_cell_fw = tf.keras.layers.LSTM(params["lstm_size"], return_sequences=True)(
        embeddings
    )
    output = tf.keras.layers.Dropout(rate=dropout)(lstm_cell_fw)
    logits = tf.keras.layers.Dense(len(tags), name="myscores")(output)

    # CRF
    transition_params = iob_transition_params(tags)

    def custom_loss(seqlen):
        def CRFLoss(y_true, y_pred):
            y_true = math_ops.cast(y_true, tf.int32)
            y_true = tf.reshape(y_true, [-1, 512])

            log_likelihood, _ = tfa.text.crf.crf_log_likelihood(
                inputs=y_pred,
                tag_indices=y_true,
                sequence_lengths=seqlen,
                transition_params=transition_params,
            )

            return tf.reduce_mean(-log_likelihood)

        return CRFLoss

    model = tf.keras.Model(
        inputs=[embeddings, nwords], outputs=logits, name="ner_simple"
    )

    model.compile(optimizer="adam", loss=custom_loss(nwords), metrics=["accuracy"])
    estimator = tf.keras.estimator.model_to_estimator(keras_model=model, model_dir=params["eval_dir"])