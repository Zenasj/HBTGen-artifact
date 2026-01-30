import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import os

from absl import app, flags, logging
import tensorflow as tf

FLAGS = flags.FLAGS

flags.DEFINE_string('workspace', None, '')


class Embeddings(tf.keras.layers.Layer):
  def __init__(self):
    super().__init__()
    with tf.device('cpu:0'):
      self.table = [
        tf.Variable(name=f'shard_{idx}', initial_value=tf.keras.initializers.glorot_normal()([int(2e6), 256]))
        for idx in range(20)
      ]

  def call(self, inputs):
    with tf.device('cpu:0'):
      looked_up = tf.reduce_sum(tf.nn.embedding_lookup(self.table, inputs), axis=1)
    return looked_up


class M(tf.keras.Model):
  def __init__(self, strategy):
    super().__init__()
    with tf.device('cpu:0'):
      self.embeddings = Embeddings()
    with strategy.scope():
      self.mlp = tf.keras.Sequential([tf.keras.layers.Dense(1)])

  def call(self, inputs):
    inputs = self.embeddings(inputs)
    out = self.mlp(inputs)
    return out


strategy = tf.distribute.MirroredStrategy()

model = M(strategy)
loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
def _replica_loss(labels, logits):
  return tf.nn.compute_average_loss(loss_fn(labels, logits), global_batch_size=10)

with tf.device('cpu:0'):
  emb_optimizer = tf.keras.optimizers.Adam()
with strategy.scope():
  mlp_optimizer = tf.keras.optimizers.Adam()


def split_variables(tv):
  from itertools import filterfalse, tee
  from tensorflow.python.distribute.values import MirroredVariable
  def partition(pred, iterable):
    'Use a predicate to partition entries into false entries and true entries'
    # partition(is_odd, range(10)) --> 0 2 4 6 8   and  1 3 5 7 9
    t1, t2 = tee(iterable)
    return list(filterfalse(pred, t1)), list(filter(pred, t2))
  return partition(lambda v: isinstance(v, MirroredVariable), tv)


def save_or_restore(save_dir: str, step: int , **to_save):
 ckpt = tf.train.Checkpoint(**to_save)
 manager = tf.train.CheckpointManager(ckpt, save_dir, max_to_keep=10)
 latest_checkpoint = tf.train.latest_checkpoint(save_dir)
 if latest_checkpoint is not None:
   logging.info(f'Restoring checkpoint: {latest_checkpoint}')
   ckpt.restore(latest_checkpoint)
 else:
   manager.save(checkpoint_number=step)


def _train_step(inputs, labels):
  with tf.GradientTape() as tape:
    logits = model(inputs)
    loss = _replica_loss(labels, logits)

  emb_var, mlp_var = split_variables(model.trainable_variables)
  emb_grad, mlp_grad = tape.gradient(loss, [emb_var, mlp_var])
  mlp_optimizer.apply_gradients(zip(mlp_grad, mlp_var))

  return loss, emb_var, emb_grad

def distribute_step(step_fn):
  @tf.function
  def _step(*step_args):
    loss, emb_var, emb_grad = strategy.run(step_fn, args=step_args)
    loss = strategy.reduce(tf.distribute.ReduceOp.SUM, loss, axis=None)
    emb_grad = strategy.reduce(tf.distribute.ReduceOp.SUM, emb_grad, axis=None)
    emb_optimizer.apply_gradients(zip(emb_grad, emb_var))
    return loss
  return _step

def main(_argv):
  save_dir = os.path.join(FLAGS.workspace, 'save_dir')

  train_step = distribute_step(_train_step)
  ds = tf.data.Dataset.from_tensors((tf.random.uniform([10, 10], minval=0, maxval=10, dtype=tf.int64), tf.random.uniform([10, 1], minval=0, maxval=1, dtype=tf.int64)))
  ds = strategy.experimental_distribute_dataset(ds)

  for example, label in ds:
    loss = train_step(example, label)
    logging.info(f'loss: {loss}')
    break

  # Save
  logging.info(f'Saving from {save_dir}')
  save_or_restore(
    save_dir=save_dir,
    step=1,
    model=model,
    cpu_optimizer=emb_optimizer,
    gpu_optimizer=mlp_optimizer,
  )

  # Restore
  logging.info(f'Restoring from {save_dir}')
  save_or_restore(
    save_dir=save_dir,
    step=1,
    model=model,
    cpu_optimizer=emb_optimizer,
    gpu_optimizer=mlp_optimizer,
  )


if __name__ == "__main__":
  app.run(main)