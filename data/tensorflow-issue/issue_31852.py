from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

@tf.function
def train_step(dist_inputs):
    def step_fn(inputs):
        features, labels = inputs

        with tf.GradientTape() as tape:
            logits = model(features)
            #print(logits)
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                        logits=logits, labels=labels)
            loss = tf.reduce_sum(cross_entropy) * (1.0 / global_batch_size)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(list(zip(grads, model.trainable_variables)))
        return cross_entropy

    per_example_losses = mirrored_strategy.experimental_run_v2(
            step_fn, args=(dist_inputs,))
    mean_loss = mirrored_strategy.reduce(
            tf.compat.v2.distribute.ReduceOp.MEAN, per_example_losses, axis=0)
    return mean_loss

reduce_sum_out = self.mirrored_strategy.reduce(
            tf.distribute.ReduceOp.SUM, per_example_losses, axis=0)

loss  = reduce_sum_out / global_batch_size

reduce_sum_out = self.mirrored_strategy.reduce(
            tf.distribute.ReduceOp.SUM, per_example_losses, axis=0)

loss  = reduce_sum_out / global_batch_size

import tensorflow as tf
import numpy as np
from tensorflow.python.client import device_lib

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']
get_available_gpus()

global_batch_size=20
mirrored_strategy = tf.distribute.MirroredStrategy()
with mirrored_strategy.scope():
  model = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=(1,))])
  optimizer = tf.keras.optimizers.SGD()
dataset = tf.data.Dataset.from_tensors(([1.], [1.])).repeat(1000).batch(
    global_batch_size)
dist_dataset = mirrored_strategy.experimental_distribute_dataset(dataset)

@tf.function
def train_step(dist_inputs):
  def step_fn(inputs):
    features, labels = inputs

    with tf.GradientTape() as tape:
      # training=True is only needed if there are layers with different
      # behavior during training versus inference (e.g. Dropout).
      logits = model(features, training=True)
      cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
          logits=logits, labels=labels)
      loss = tf.reduce_sum(cross_entropy) * (1.0 / global_batch_size)
      #print("loss", loss)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(list(zip(grads, model.trainable_variables)))
    return cross_entropy

  per_example_losses = mirrored_strategy.experimental_run_v2(
      step_fn, args=(dist_inputs,))
  
  mean_loss = mirrored_strategy.reduce(
      tf.distribute.ReduceOp.MEAN, per_example_losses, axis=0)
  #print("mean_loss", mean_loss)
  return mean_loss

with mirrored_strategy.scope():
  for inputs in dist_dataset:
    print(train_step(inputs))