import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf
print('tf version', tf.__version__)
import os

SIG=[tf.TensorSpec([None], tf.int32)]

class CustomModel(tf.keras.models.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.embedding = tf.keras.layers.Embedding(input_dim=3, output_dim=4, mask_zero=True)

    def call(self, x):
        _x = tf.reduce_sum(self.embedding(x))   # <<<< new!
        _x = tf.random.normal((3,)) + _x 
        return _x

class CustomModule(tf.Module):
    def __init__(self):
        self.model = CustomModel()

    @tf.function(input_signature=SIG)
    def f1(self, x):
        print('tracing f1')
        return self.model(x)


module = CustomModule()
# added CustomModel.embedding ==> module.f1 now traced 2 times instead of once (same in TF2.3 and TF2.4)
print('======== pretty_printed_concrete_signatures() ========\n', module.f1.pretty_printed_concrete_signatures())  

# with tf.saved_model.save(), TF2.4 does an extra tracing
print('======= tf.saved_model.save() =======')
module = CustomModule()
os.makedirs('saved_model', exist_ok=True)
signatures = {"serving_default": module.f1.get_concrete_function(*SIG)}
tf.saved_model.save(module, 'saved_model', signatures=signatures) # module.f1() traced 3 times with TF2.4, 2 times with TF2.3