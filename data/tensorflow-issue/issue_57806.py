import shutil
import tensorflow as tf


class PrimaryModule(tf.Module):
    def __init__(self, name=None):
        super(PrimaryModule, self).__init__(name=name)
        initializer = tf.lookup.TextFileInitializer(
            "chzhu_vocab.txt",
            key_dtype=tf.int64,
            key_index=0,
            value_dtype=tf.int64,
            value_index=1,
            delimiter=" ",
        )
        self.table = tf.lookup.StaticVocabularyTable(initializer, 1)

    @tf.function
    def __call__(self, inputs):
        return self.table.lookup(inputs)


model = PrimaryModule()
print(model(tf.constant([509323409], dtype=tf.int64)))
tf.saved_model.save(model, "asset_model")
imported_new = tf.saved_model.load("asset_model")
print(imported_new(tf.constant([509323409], dtype=tf.int64)))
tf.saved_model.save(imported_new, "wrapped_assets_model")

# Error when loading the model for the second time, it will be trying to find the original path
# instead of path in the asset folder inside model
shutil.rmtree("asset_model")
shutil.rm("chzhu_vocab.txt")
tf.saved_model.load("wrapped_assets_model")

import tensorflow as tf

class PrimaryModule(tf.Module):
    def __init__(self, name=None):
        super(PrimaryModule, self).__init__(name=name)
        self.initializer = tf.lookup.TextFileInitializer(
            "chzhu_vocab.txt",
            key_dtype=tf.int64,
            key_index=0,
            value_dtype=tf.int64,
            value_index=1,
            delimiter=" ",
        )
        self.table = tf.lookup.StaticVocabularyTable(self.initializer, 1)

    @tf.function
    def __call__(self, inputs):
        return self.table.lookup(inputs)

# ..same saving logics

import shutil
import tensorflow as tf


class PrimaryModule(tf.Module):
    def __init__(self, name=None):
        super(PrimaryModule, self).__init__(name=name)
        initializer = tf.lookup.TextFileInitializer(
            "chzhu_vocab.txt",
            key_dtype=tf.int64,
            key_index=0,
            value_dtype=tf.int64,
            value_index=1,
            delimiter=" ",
        )
        self.table = tf.lookup.StaticVocabularyTable(initializer, 1)

    @tf.function
    def __call__(self, inputs):
        return self.table.lookup(inputs)


model = PrimaryModule()
print(model(tf.constant([509323409], dtype=tf.int64)))
tf.saved_model.save(model, "asset_model")
imported_new = tf.saved_model.load("asset_model")
print(imported_new(tf.constant([509323409], dtype=tf.int64)))
tf.saved_model.save(imported_new, "wrapped_assets_model")

# Error when loading the model for the second time
shutil.rmtree("asset_model")
tf.saved_model.load("wrapped_assets_model")