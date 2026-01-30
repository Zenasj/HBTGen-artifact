import random
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
import numpy as np

ipt = Input((16,))
out = Dense(16)(ipt)
model = Model(ipt, out)
model.compile('adam', 'mse')

x = np.random.randn(32, 16)
model.train_on_batch(x, x)

grads = model.optimizer.get_gradients(model.total_loss, model.layers[-1].output)
grads_fn = K.function(inputs=[model.inputs[0], model._feed_targets[0], K.learning_phase()], 
                      outputs=grads)

import tensorflow.keras.backend as K
from tensorflow.python.eager import context
from tensorflow.python.ops import array_ops
import weakref
from tensorflow.python.framework import func_graph

def symbolic_learning_phase():
  graph = get_graph()
  with graph.as_default():
    if graph not in _GRAPH_LEARNING_PHASES:
      with K.name_scope(''):
        phase = array_ops.placeholder_with_default(
            False, shape=(), name='keras_learning_phase')
      _GRAPH_LEARNING_PHASES[graph] = phase
    return _GRAPH_LEARNING_PHASES[graph]

def get_graph():
  if context.executing_eagerly():
    global _GRAPH
    if _GRAPH is None:
      _GRAPH = func_graph.FuncGraph('keras_graph')
    return _GRAPH
  else:
    return ops.get_default_graph()

_GRAPH = None
_GRAPH_LEARNING_PHASES = weakref.WeakKeyDictionary()

symbolic_learning_phase()
# <tf.Tensor 'keras_learning_phase:0' shape=() dtype=bool>