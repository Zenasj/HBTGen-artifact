import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

class EmbeddingMerger(tf.keras.layers.Layer):
  def __init__(self, list_features, **kwargs):
    super().__init__(**kwargs)
    self._supports_ragged_inputs = True
    self.embeddings = {feature: Embedding(10, 3) for feature in list_features}
    self.mean = tf.keras.layers.Lambda(tf.reduce_mean, arguments=dict(axis=1))
  def call(self, inputs):
    tensors = [self.embeddings[col](inputs[col]) for col in inputs]
    tensors = [self.mean(inp) for inp in tensors]
    return Add()(tensors)

list_features = ['feature_1', 'feature_2']
feature_1 = tf.ragged.constant([[0], [1, 3]])
feature_2 = tf.ragged.constant([[1, 2], [4]])
f = {'feature_1': feature_1,
     'feature_2': feature_2}
f_inputs = {'feature_1': Input(shape=(), name='feature_1', ragged=True),
            'feature_2': Input(shape=(), name='feature_2', ragged=True)}
out = EmbeddingMerger(list_features)(f_inputs)
model = Model(f_inputs, out)

truth = model.predict(f)
truth

model.save('/tmp/test')
model_reloaded = tf.keras.models.load_model('/tmp/test')
model_reloaded.predict(f)

f_inputs = {'feature_1': Input(shape=[None], name='feature_1', ragged=True),
            'feature_2': Input(shape=[None], name='feature_2', ragged=True)}