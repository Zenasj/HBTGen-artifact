from tensorflow import keras
from tensorflow.keras import layers

import os
import shutil
import tensorflow as tf

class Model(tf.keras.layers.Layer):

    def __init__(self, vocabulary_path):
        super(Model, self).__init__()
        initializer = tf.lookup.TextFileInitializer(
            vocabulary_path,
            tf.string,
            tf.lookup.TextFileIndex.WHOLE_LINE,
            tf.int64,
            tf.lookup.TextFileIndex.LINE_NUMBER)
        self.table = tf.lookup.StaticVocabularyTable(initializer, num_oov_buckets=1)
        #self.table = tf.lookup.StaticHashTable(initializer, 0)

    def call(self, tokens):
        return self.table.lookup(tokens)

    @tf.function(input_signature=(tf.TensorSpec([None], dtype=tf.string),))
    def serve(self, tokens):
        return self(tokens)


vocabulary_path = "/tmp/vocab.txt"
with open(vocabulary_path, "w") as vocabulary_file:
    vocabulary_file.write("a\nb\nc\n")

model = Model(vocabulary_path)

export_dir = "/tmp/model"
if os.path.exists(export_dir):
    shutil.rmtree(export_dir)
tf.saved_model.save(model, export_dir, signatures=model.serve)
assets = os.listdir(os.path.join(export_dir, "assets"))
assert len(assets) == 1