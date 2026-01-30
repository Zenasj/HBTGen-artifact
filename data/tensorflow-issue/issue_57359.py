import tensorflow as tf  # type: ignore
from tensorflow import keras
from keras import layers  # type: ignore
from keras import backend as K
physical_devices = tf.config.list_physical_devices("GPU")
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.333
session = tf.compat.v1.Session(config=config)
K.set_session(session)

tf.config.experimental.enable_tensor_float_32_execution(False)

tf.config.experimental.enable_tensor_float_32_execution(False)
for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)

from sklearn.base import BaseEstimator, TransformerMixin
import tensorflow_hub as hub

class UseEmbedder(TransformerMixin, BaseEstimator):
    def __init__(self):
        self._embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
        
    def fit(self, X, y=None, sample_weight=None):
        return self
    
    def transform(self, X):
        return self._embed(X).numpy()
    
    def fit_transform(self, X, y=None, sample_weight=None):
        return self.transform(X)


embedding_transformer = UseEmbedder()
embedding_transformer.transform(['why did this just break'])