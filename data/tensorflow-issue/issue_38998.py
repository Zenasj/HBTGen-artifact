import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class EmbeddingMean(tf.keras.layers.Layer):
  def __init__(self, **kwargs):
    super(EmbeddingMean, self).__init__()
    self._supports_ragged_inputs = True
  def call(self, inputs, **kwargs):
    return tf.reduce_mean(inputs, axis=1)

feature = Input(shape=(None,), ragged=True, name='input_1', dtype=tf.int32)
embedded = Embedding(10, 3)(feature)
embedded_mean = EmbeddingMean()(embedded)
m = Model(feature, Dense(1)(embedded_mean))