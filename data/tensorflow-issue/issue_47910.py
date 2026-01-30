import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import numpy as np
import tensorflow as tf

from tensorflow.python.ops import math_ops


@tf.keras.utils.register_keras_serializable()
class BucketizeLayer(tf.keras.layers.experimental.preprocessing.PreprocessingLayer):
    
    def __init__(self, quantiles, **kwargs):
        self.quantiles = quantiles
        super(BucketizeLayer, self).__init__(**kwargs)
        self._boundaries = None
        
    def adapt(self, data):
        if isinstance(data, tf.Tensor):
            data = data.numpy()
            
        self._boundaries = np.nanquantile(data, self.quantiles).tolist()
        
    def call(self, data):
        return math_ops.bucketize(data, self._boundaries)
    
    def get_config(self):
        config = {'quantiles': self.quantiles}
        base_config = super(BucketizeLayer, self).get_config()
        return dict(**config, **base_config)


data = 3 * np.random.randn(100) + 5
bucketize = BucketizeLayer(quantiles=[0.1, 0.5, 0.9])
bucketize.adapt(data)
print(bucketize(data))


inp = tf.keras.Input(shape=(1,), dtype=tf.float32)
model = tf.keras.models.Model(inp, bucketize(inp))
model.save('bucketize/')
del model
model = tf.keras.models.load_model('bucketize/')
model.layers[1]._boundaries
# output: ListWrapper([])