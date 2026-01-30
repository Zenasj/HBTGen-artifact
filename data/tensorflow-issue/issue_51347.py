import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

sex = tf.keras.Input(shape=(1,), name="sex", dtype=tf.string)
x = tf.feature_column.categorical_with_vocabulary_file("sex","./sex.txt", num_oov_bucuckets=5)
x = tf.keras.layers.DenseFeatures([tf.feature_column.embedding_column(x, 2)])
out = x({"sex",sex})
model = tf.keras.Model(inputs=sex, outputs=out)
tf.save_model.save(model, "./no_assets_model")

sex = tf.keras.Input(shape=(1,), name="sex", dtype=tf.string)
x = tf.feature_column.categorical_with_vocabulary_file("sex","./sex.txt")
x = tf.keras.layers.DenseFeatures([tf.feature_column.embedding_column(x, 2)])
out = x({"sex",sex})
model = tf.keras.Model(inputs=sex, outputs=out)
tf.save_model.save(model, "./has_assets_model")