from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

import os
import shutil
import tensorflow as tf

current_dir = os.path.realpath(os.path.dirname(__file__))
save_dir = os.path.join(current_dir, 'testhop')
save_dir_wbracket = os.path.join(current_dir, 'test[hop')


if os.path.isdir(save_dir):
    shutil.rmtree(save_dir)
os.mkdir(save_dir)
assert os.path.isdir(save_dir)

if os.path.isdir(save_dir_wbracket):
    shutil.rmtree(save_dir_wbracket)
os.mkdir(save_dir_wbracket)
assert os.path.isdir(save_dir_wbracket)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(
        16, batch_input_shape=[256, 1], activation='relu'),
    tf.keras.layers.Dense(16, activation='tanh'),
    tf.keras.layers.Dense(1),
])

lr = tf.Variable(1.)
reg_param = tf.Variable(1.)
optim = tf.keras.optimizers.SGD(lr)

ckpt = tf.train.Checkpoint(
    model=model,
    lr=lr,
    reg_param=reg_param,
)

manager = tf.train.CheckpointManager(ckpt, save_dir, max_to_keep=1)
manager_wbracket = tf.train.CheckpointManager(ckpt, save_dir_wbracket, max_to_keep=1)

manager.save()
assert os.path.isfile(os.path.join(save_dir, 'checkpoint'))
manager_wbracket.save()
assert os.path.isfile(os.path.join(save_dir_wbracket, 'checkpoint'))

# Works
print(manager.latest_checkpoint)
ckpt.restore(manager.latest_checkpoint)

# Does not work
print(manager_wbracket.latest_checkpoint)
ckpt.restore(manager_wbracket.latest_checkpoint)