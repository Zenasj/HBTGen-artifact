from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

def variable_creator(**kwargs):
    kwargs["use_resource"] = False
    return variable_scope.default_variable_creator(None, **kwargs)
getter = lambda next_creator, **kwargs: variable_creator(**kwargs)
with variable_scope.variable_creator_scope(getter):
    model = ...

import tensorflow as tf
import numpy as np
from tensorflow.python.ops import variable_scope

def serialize_graph(model):
    g = tf.graph_util.convert_variables_to_constants(
            tf.keras.backend.get_session(),
            tf.keras.backend.get_session().graph.as_graph_def(),
            #[n.name for n in tf.keras.backend.get_session().graph.as_graph_def().node],
            [t.op.name for t in model.outputs]
            )
    return g

def build_save_restore(model):
    model.compile('sgd', loss='mse')
    model.fit(np.array([[1]]),np.array([[1]]), verbose=0)
    gdef = serialize_graph(model)
    newg = tf.Graph()
    with newg.as_default():
        tf.import_graph_def(gdef)
        print("*"*25)
        print("restored successfully")
        print("*"*25)

def variable_creator(**kwargs):
    kwargs["use_resource"] = False
    return variable_scope.default_variable_creator(None, **kwargs)

model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(1)
        ])
build_save_restore(model)

getter = lambda next_creator, **kwargs: variable_creator(**kwargs)
with variable_scope.variable_creator_scope(getter):
    model = tf.keras.models.Sequential([
            tf.keras.layers.Embedding(1, 1, input_length=1),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1),
            ])
    build_save_restore(model)