from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

# Tested with TF 2.0.0, Linux Ubuntu 18.04, Python 3.7.3, TF installed from binary
import tensorflow as tf

# create model and optimizer and checkpoint
model = tf.keras.models.Sequential([tf.keras.layers.Dense(5)])
opt = tf.keras.optimizers.Adam(0.1)
checkpoint_dir = 'ckpts'
ckpt = tf.train.Checkpoint(opt=opt, model=model)
manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=3)

# train with one example
example_x = tf.constant([[1.]])
example_y = tf.constant([[1.,2.,3.,4.,5.]])
model.compile(loss="mean_squared_error", optimizer=opt)
model.fit(example_x, example_y, epochs=1)

save_path = manager.save()
print("Saved checkpoint: {}".format(save_path))

# ========== restart from scratch but restore from checkpoint
model = tf.keras.models.Sequential([tf.keras.layers.Dense(5)])
opt = tf.keras.optimizers.Adam(0.1)

ckpt = tf.train.Checkpoint(opt=opt, model=model)
manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=3)
print('restoring...')
status = ckpt.restore(manager.latest_checkpoint)
# assert_consumed() fails with:
    # AssertionError: Unresolved object in checkpoint (root).opt.iter: attributes {
    #   name: "VARIABLE_VALUE"
    #   full_name: "Adam/iter"
    #   checkpoint_key: "opt/iter/.ATTRIBUTES/VARIABLE_VALUE"
status.assert_consumed()