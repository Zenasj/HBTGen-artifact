import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

fea['hisList'] = tf.io.VarLenFeature(tf.int64)
tmp = tf.feature_column.categorical_column_with_hash_bucket('hisList', 1000, dtype=tf.string)
emb = tf.feature_column.embedding_column(tmp , 12)
feature_layer = tf.keras.layers.DenseFeatures([emb])

model = tf.keras.Sequential([
      feature_layer,
      tf.keras.layers.Dense(64, activation='relu'),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dense(1, activation='sigmoid')
    ])