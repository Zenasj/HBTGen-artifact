from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

full_model.layers[1].get_config()

Out[9]: {'name': 'string_lookup',
 'trainable': True,
 'dtype': 'int64',
 'invert': False,
 'max_tokens': None,
 'num_oov_indices': 1,
 'oov_token': '[UNK]',
 'mask_token': None,
 'output_mode': 'int',
 'sparse': False,
 'pad_to_max_tokens': False,
 'vocabulary': ListWrapper(['a', 'b']),
 'idf_weights': None,
 'encoding': 'utf-8'}

full_model_loaded.layers[1].get_config()

Out[10]: {'name': 'string_lookup',
 'trainable': True,
 'dtype': 'int64',
 'invert': False,
 'max_tokens': None,
 'num_oov_indices': 1,
 'oov_token': '[UNK]',
 'mask_token': None,
 'output_mode': 'int',
 'sparse': False,
 'pad_to_max_tokens': False,
 'vocabulary': ListWrapper([]),
 'idf_weights': None,
 'encoding': 'utf-8'}

@tf.keras.utils.register_keras_serializable()
class MyStringLookup(tf.keras.layers.StringLookup):
    def get_config(self):
        base_config = super().get_config()
        custom = {"vocabulary": self.get_vocabulary()}
        return {**base_config, **custom}

import tensorflow as tf
import pickle

model_input = tf.keras.Input(shape=(1,), dtype=tf.int64)
lookup = tf.keras.layers.StringLookup(vocabulary=['a', 'b'])(model_input)
output = tf.keras.layers.Dense(10)(lookup)
full_model = tf.keras.Model(model_input, output)

# this part should work
model_bytes = pickle.dumps(full_model)
model_recovered = pickle.loads(model_bytes)


# this part should throw an error
full_model.save("/tmp/temp_model")
full_model_loaded = tf.keras.models.load_model("/tmp/temp_model")
model_bytes_2 = pickle.dumps(full_model_loaded)
model_recovered_2 = pickle.loads(model_bytes_2)

import tensorflow as tf
import pickle

model_input = tf.keras.Input(shape=(1,), dtype=tf.int64)
lookup = tf.keras.layers.StringLookup(vocabulary=['a', 'b'])(model_input)
lookup = tf.keras.layers.Flatten()(lookup)
output = tf.keras.layers.Dense(10)(lookup)
full_model = tf.keras.Model(model_input, output)

# this part should work
model_bytes = pickle.dumps(full_model)
model_recovered = pickle.loads(model_bytes)


# this part should throw an error
full_model.save("/tmp/temp_model")
full_model_loaded = tf.keras.models.load_model("/tmp/temp_model")
model_bytes_2 = tf.keras.layers.serialize(full_model_loaded)
model_recovered_2 = tf.keras.layers.deserialize(model_bytes_2)

import tensorflow as tf
import pickle

@tf.keras.utils.register_keras_serializable()
class MyStringLookup(tf.keras.layers.StringLookup):
    def get_config(self):
        base_config = super().get_config()
        custom = {"vocabulary": self.get_vocabulary()}
        return {**base_config, **custom}

model_input = tf.keras.Input(shape=(1,), dtype=tf.int64)
lookup = MyStringLookup(vocabulary=['a', 'b'])(model_input)
lookup = tf.keras.layers.Flatten()(lookup)
output = tf.keras.layers.Dense(10)(lookup)
full_model = tf.keras.Model(model_input, output)

# this part should work
model_bytes = pickle.dumps(full_model)
model_recovered = pickle.loads(model_bytes)


# this part should throw an error
full_model.save("/tmp/temp_model")
full_model_loaded = tf.keras.models.load_model("/tmp/temp_model")
model_bytes_2 = tf.keras.layers.serialize(full_model_loaded)
model_recovered_2 = tf.keras.layers.deserialize(model_bytes_2)

import tensorflow as tf
import pickle

tf.keras.utils.get_custom_objects().clear()


@tf.keras.utils.register_keras_serializable()
class MyStringLookup(tf.keras.layers.StringLookup):
    def get_config(self):
        base_config = super().get_config()
        vocabulary = self.get_vocabulary()
        custom = {"vocabulary": vocabulary}
        return {**base_config, **custom}


def my_clone_function(layer):
    if isinstance(layer, tf.keras.layers.StringLookup):
        clone_layer = MyStringLookup(vocabulary=layer.get_vocabulary())
        return clone_layer
    return layer


# create model
model_input = tf.keras.Input(shape=(1,), dtype=tf.int64)
lookup = tf.keras.layers.StringLookup(vocabulary=['a', 'b'])(model_input)
output = tf.keras.layers.Dense(10)(lookup)
model = tf.keras.Model(model_input, output)

# model StringLookup has vocabulary set
print('\ncorrect:\n')
print(model.layers[1].get_config()['vocabulary'])
# ListWrapper(['a', 'b'])

# save model
model.save("/tmp/temp_model")

# load model
model_orginal_reloaded = tf.keras.models.load_model("/tmp/temp_model")

# model StringLookup now has no vocabulary set
print('\nbroken:\n')
print(model_orginal_reloaded.layers[1].get_config()['vocabulary'])
# ListWrapper([])

# fix the model
model_fixed = tf.keras.models.clone_model(
    model,
    clone_function=my_clone_function
)

# we now see a correct model vocab in the MyStringLookup layer
print('\nfixed:\n')
print(model_fixed.layers[1].get_config()['vocabulary'])