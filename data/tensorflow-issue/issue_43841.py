from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf
from tensorflow.python import keras as pykeras
import weakref
import copy

initial_builing = True
model_loading = True

if initial_builing:

    layer1 = tf.keras.Input((1,),)
    layer2 = tf.keras.layers.Dense(1)

    model_output = layer2(layer1)[:,:-1]

    model = tf.keras.Model(layer1, model_output)
    model.summary()
    model.save('testmodel')
    cache_obj_name_uids = copy.copy(pykeras.backend.PER_GRAPH_OBJECT_NAME_UIDS)
    tf.keras.backend.reset_uids()
if model_loading:
    model = tf.keras.models.load_model('testmodel')
    pykeras.backend.PER_GRAPH_OBJECT_NAME_UIDS=  cache_obj_name_uids
    model = tf.keras.Model(model.inputs, model.layers[-1].output[:,:-1])
    model.summary()

import tensorflow as tf
from tensorflow.python import keras as pykeras
import weakref
from tensorflow.python.framework import ops
import copy
import gc

initial_builing = True
model_loading = True
cache_obj_name_uids = None
if initial_builing:

    layer1 = tf.keras.Input((1,),)
    layer2 = tf.keras.layers.Dense(1)

    model_output = layer2(layer1)[:,:-1]

    model = tf.keras.Model(layer1, model_output)
    model.summary()
    model.save('testmodel')
    cache_obj_name_uids = copy.copy(pykeras.backend.PER_GRAPH_OBJECT_NAME_UIDS[ops.get_default_graph()])
    del model, model_output
    tf.keras.backend.clear_session()
    gc.collect()
if model_loading:
    model = tf.keras.models.load_model('testmodel')
    pykeras.backend.PER_GRAPH_OBJECT_NAME_UIDS[ops.get_default_graph()]=  cache_obj_name_uids
    model = tf.keras.Model(model.inputs, model.layers[-1].output[:,:-1])
    model.summary()