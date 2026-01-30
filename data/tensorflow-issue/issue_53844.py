from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import numpy as np
import tensorflow as tf

# build the initial model
x = tf.keras.layers.Input((50,))
out = tf.keras.layers.Dense(1, activation='sigmoid')(x)
model = tf.keras.Model(x, out)

# setup an optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
opt_path = 'opt_wights.npy'
grad_vars = model.trainable_weights
zero_grads = [tf.zeros_like(w) for w in grad_vars]
optimizer.apply_gradients(zip(zero_grads, grad_vars))

# save its state
opt_weights = optimizer.get_weights()
np.save(opt_path, [np.int32(opt_weights[0])] + opt_weights[1:])

# create the distributed strategy
cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver('')
tf.config.experimental_connect_to_cluster(cluster_resolver)
tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
strategy = tf.distribute.TPUStrategy(cluster_resolver)

# build the model and optimizer again
with strategy.scope():
  x = tf.keras.layers.Input((50,))
  out = tf.keras.layers.Dense(1, activation='sigmoid')(x)
  optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
  model = tf.keras.Model(x, out)
  model.compile(optimizer=optimizer)

# set up the weights for the model
with strategy.scope():
  grad_vars = model.trainable_weights
  zero_grads = [tf.zeros_like(w) for w in grad_vars]
  opt_weights = np.load(opt_path, allow_pickle=True)
  opt_weights = [tf.constant(w) for w in opt_weights]

@tf.function
def _model_weight_setting():
  optimizer.apply_gradients(zip(zero_grads, grad_vars))

# apply the gradients to the model
strategy.run(_model_weight_setting)

# update the weights of the model
with strategy.scope():
  optimizer.set_weights(opt_weights)