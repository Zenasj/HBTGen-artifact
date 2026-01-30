from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
import tensorflowjs as tfjs

gpus = tf.config.experimental.list_physical_devices('GPU')
assert len(gpus) > 1, 'Probably multiple GPUs required'
tf.config.experimental.set_visible_devices([gpus[0]], 'GPU')

class EmbModule(tf.Module):
    
    def __init__(self):
        super().__init__()
    
    @tf.function(input_signature = [tf.TensorSpec(shape = [1, 2], dtype = tf.int32)])
    def apply(self, inp):
        return tf.nn.embedding_lookup(tf.ones([3, 4]), inp)

embModule = EmbModule()

tf.saved_model.save(embModule, 'embModule/1/')
# Next line fails as single GPU (of few in the system) is used:
tfjs.converters.convert_tf_saved_model('embModule/1/', 'embModule/1-js')

import tensorflow as tf
import tensorflowjs as tfjs

class EmbModule(tf.Module):
    
    def __init__(self):
        super().__init__()
        self.emb = tf.keras.layers.Embedding(3, 4)
        self.emb.build((1, 2))
    
    @tf.function(input_signature = [tf.TensorSpec(shape = [1, 2], dtype = tf.int32)])
    def apply(self, inp):
        return self.emb(inp)

embModule = EmbModule()

tf.saved_model.save(embModule, 'embModule/1/')
# Next line fails as 'Embedding' layer was built in constructor:
tfjs.converters.convert_tf_saved_model('embModule/1/', 'embModule/1-js')