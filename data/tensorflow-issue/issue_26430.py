from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(256)
])

x = tf.zeros([10, 3])   # dummy input
model(x)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

# finalize the graph (or it's done automatically in MonitoredSession)
tf.get_default_graph().finalize()

# Error!
model.save('/tmp/keras-test')

is_initialized = session.run(
     [variables_module.is_variable_initialized(v) for v in candidate_vars])

# g : tf.Graph (e.g. tf.get_default_graph())

g._finalized = False       # de-finalized for a while
# ... (add more ops) ...
g.finalize()              # finalize again