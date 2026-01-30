import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class EmbeddingMerger(tf.keras.layers.Layer):
  def __init__(self, list_features, **kwargs):
    super().__init__(**kwargs)
    self.embeddings = {feature: Embedding(10, 3) for feature in list_features}
  def call(self, inputs):
    tensors = [self.embeddings[col](inputs[col]) for col in inputs]
    return Add()(tensors)

list_features = ['feature_1', 'feature_2']
feature_1 = tf.constant([0, 1, 3])
feature_2 = tf.constant([1, 2, 4])
f = {'feature_1': feature_1,
     'feature_2': feature_2}
f_inputs = {'feature_1': Input(shape=(), name='feature_1'),
            'feature_2': Input(shape=(), name='feature_2')}
out = EmbeddingMerger(list_features)(f_inputs)
model = Model(f_inputs, out)