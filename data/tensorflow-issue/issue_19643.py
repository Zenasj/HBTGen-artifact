from tensorflow import keras
from tensorflow.keras import layers

#!/usr/bin/python
import tensorflow as tf

graph = tf.get_default_graph()
tf.keras.backend.set_learning_phase(True)
features = tf.zeros(shape=(3, 10), dtype=tf.float32)
normed = tf.keras.layers.BatchNormalization()(features)

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
print('n_ops:        %d' % len(graph.get_operations()))
print('n_update_ops: %d' % len(update_ops))

updates = layer.updates

updates = model.updates

import tensorflow as tf
x = tf.zeros(shape=(2, 3), dtype=tf.float32)
y = tf.keras.layers.Dense(4)(x)
print('variables: %d' % len(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)))

import tensorflow as tf

tf.keras.backend.set_learning_phase(True)
input_shape = (3,)
inp = tf.keras.Input(shape=input_shape, dtype=tf.float32)
x = tf.keras.layers.Dense(4, input_shape=input_shape)(inp)
x = tf.keras.layers.BatchNormalization()(x, training=True)

model = tf.keras.Model(inp, x)
model.compile(tf.train.AdamOptimizer(1e-3), 'mean_squared_error')
print('model_updates: %d' % len(model.updates))
for update in model.updates:
    print(update.name, update._unconditional_update)  # False

estimator = tf.keras.estimator.model_to_estimator(model)

z = tf.zeros((2, 3), dtype=tf.float32)
labels = tf.zeros((2, 4), dtype=tf.float32)

spec = estimator.model_fn(z, labels, mode='train', config=None)

print(len(tf.get_collection(tf.GraphKeys.UPDATE_OPS)))  # 0