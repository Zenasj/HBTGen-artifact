from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf

# Create Simple Signature Function
def outer_fn(model, vocab_file, num_oov=1):

    model.init = tf.lookup.TextFileInitializer(
            filename=vocab_file,
            key_dtype=tf.string,
            key_index=tf.lookup.TextFileIndex.WHOLE_LINE,
            value_dtype=tf.int64,
            value_index=tf.lookup.TextFileIndex.LINE_NUMBER
        )
    model.table = tf.lookup.StaticVocabularyTable(model.init, num_oov, lookup_key_dtype=tf.string)

    @tf.function
    def inner_fn(b):
        return model.table.lookup(b)

    return inner_fn


def out_intent_fn(model, vocab_file, num_oov=1):

    init = tf.lookup.TextFileInitializer(
                filename=vocab_file,
                key_dtype=tf.string,
                key_index=tf.lookup.TextFileIndex.WHOLE_LINE,
                value_dtype=tf.int64,
                value_index=tf.lookup.TextFileIndex.LINE_NUMBER)
    model.table = tf.lookup.StaticVocabularyTable(init, num_oov, lookup_key_dtype=tf.string)

    @tf.function
    def predict_fn(probes):
        ptokens = model.table.lookup(probes)
        return ptokens

    return predict_fn

def outmost():
    export_path = "./"
    vocab_file = "./test.file"

    model = tf.keras.Sequential([
        tf.keras.Input(shape=(1,), dtype=tf.float32),
        tf.keras.layers.Dense(1),
        tf.keras.layers.Dropout(rate=0.2)
    ])

    # Signature
    signatures = {
        'service_default':
            out_intent_fn(
                model, vocab_file=vocab_file
                ).get_concrete_function(tf.TensorSpec(shape=[None], dtype=tf.string))
    }

    # Model Save [ERROR - Untracked Tensor]
    model.save(export_path, signatures=signatures)

if name == "main":
    outmost()