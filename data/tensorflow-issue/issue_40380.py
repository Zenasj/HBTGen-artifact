import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers

old_opt = model.optimizer
# it WORKS, you have to assign a object which a sub-class of tf.keras.optimizers.Optimizer
model.optimizer = tf.keras.optimizers.SGD()
model.save(save_to, include_optimizer=False)
model.optimizer = old_opt

old_opt = model.optimizer
# it NOT works!
model.optimizer = None
model.save(save_to, include_optimizer=False)
model.optimizer = old_opt