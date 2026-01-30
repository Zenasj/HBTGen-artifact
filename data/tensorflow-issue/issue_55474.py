import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

strategy = tf.distribute.MirroredStrategy()
ds = tf.data.Dataset.list_files(".*", shuffle=False).map(lambda x: (tf.strings.length(x), tf.strings.length(x)))
with strategy.scope():
  dummy_model = tf.keras.Sequential()
  dummy_model.add(tf.keras.layers.Dense(1, input_shape=(1,)))
  dummy_model.compile(loss="mse")
dummy_model.fit(ds.batch(4))