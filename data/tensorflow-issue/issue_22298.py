from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf

from tensorflow.python.ops import variable_scope as vs

net = tf.keras.Sequential([
    tf.keras.layers.Dense(10, name='fc1')
])

x = tf.random_uniform([10, 3])
y = net(x)

# all variable are created 
vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
for v in vars:
    print(v.name)

# but variable scope doesn't see them
# then no variable to initialize for tf.train.init_from_checkpoint
var_store = vs._get_default_variable_store()._vars
print(var_store)