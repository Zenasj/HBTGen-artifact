import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import os
import tensorflow as tf
import numpy as np
import os.path as osp
import pickle
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model

#build model
X, y = np.random.rand(100, 50), np.random.randint(2, size=100)
x = Input((50,))
out = Dense(1, activation='sigmoid')(x)
model = Model(x, out)

#build setup an optimizer and save the state
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
opt_path = 'opt_wights.npy'
grad_vars = model.trainable_weights
zero_grads = [tf.zeros_like(w) for w in grad_vars]
optimizer.apply_gradients(zip(zero_grads, grad_vars))
np.save(opt_path, optimizer.get_weights())

#do the same thing with a load, but in strategy scope
#get the strategy
bTPU = True
if bTPU:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.TPUStrategy(tpu)
else:    
    gpus = tf.config.experimental.list_logical_devices("GPU")
    if len(gpus) > 1:
        strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
    elif len(gpus) == 1:
        strategy = tf.distribute.get_strategy()
    else:
        strategy = tf.distribute.get_strategy()

with strategy.scope():
    #build the model and optimizer again
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
    X, y = np.random.rand(100, 50), np.random.randint(2, size=100)
    x = tf.keras.layers.Input((50,))
    out = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.Model(x, out)
    @tf.function	
    def _model_weight_setting():
        opt_weights = np.load(opt_path, allow_pickle=True)
        grad_vars = model.trainable_weights
        zero_grads = [tf.zeros_like(w) for w in grad_vars]
        optimizer.apply_gradients(zip(zero_grads, grad_vars))
        optimizer.set_weights(opt_weights)
    strategy.run(_model_weight_setting)