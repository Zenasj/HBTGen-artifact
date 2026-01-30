from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import os
import tensorflow as tf
import numpy as np
import json

def mnist_dataset():
    '''
    Load mnist dataset. No batching
    '''
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    x_train = x_train / np.float32(255)
    y_train = y_train.astype(np.int64)
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(60000)
    return train_dataset

class CustomModel(tf.keras.Model):
    def __init__(self, model):
        super(CustomModel, self).__init__()
        self.model = model
        self.loss_tracker= tf.keras.metrics.Mean(name='loss')
        
    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            # Forward pass
            y_pred = self.model(x, training=True)
            # Compute our own loss. Shape (batch_size,)
            loss = tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits = True, reduction=tf.keras.losses.Reduction.NONE)(y, y_pred)
           
        # Compute gradients
        trainable_vars = self.model.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics
        self.loss_tracker.update_state(loss)
        self.compiled_metrics.update_state(y, y_pred)
        metrics = {m.name: m.result() for m in self.metrics}
        return metrics

def build_and_compile_cnn_model():
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(28, 28)),
        tf.keras.layers.Reshape(target_shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)])
    model = CustomModel(model = model)
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.001))
    return model

# TRAIN
per_worker_batch_size = 32
num_workers = len(tf_config['cluster']['worker'])
global_batch_size = per_worker_batch_size * num_workers
multi_worker_dataset = mnist_dataset().batch(global_batch_size)

with strategy.scope():
  # Model building/compiling need to be within `strategy.scope()`.
  multi_worker_model = build_and_compile_cnn_model()

multi_worker_model.fit(multi_worker_dataset, epochs=3, steps_per_epoch=70)

@tf.function
def train_step(iterator):
  """Training step function."""

  def step_fn(inputs):
    """Per-Replica step function."""
    x, y = inputs
    with tf.GradientTape() as tape:
      predictions = multi_worker_model(x, training=True)
      per_batch_loss = tf.keras.losses.SparseCategoricalCrossentropy(
          from_logits=True,
          reduction=tf.keras.losses.Reduction.NONE)(y, predictions)
      loss = tf.nn.compute_average_loss(
          per_batch_loss, global_batch_size=global_batch_size)

    grads = tape.gradient(loss, multi_worker_model.trainable_variables)
    optimizer.apply_gradients(
        zip(grads, multi_worker_model.trainable_variables))
    train_accuracy.update_state(y, predictions)
    return loss

  per_replica_losses = strategy.run(step_fn, args=(next(iterator),))
  return strategy.reduce(
      tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

with strategy.scope():
  # Set reduction to `NONE` so you can do the reduction afterwards and divide by
  # global batch size.
  loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
      from_logits=True,
      reduction=tf.keras.losses.Reduction.NONE)
  def compute_loss(labels, predictions, model_losses):
    per_example_loss = loss_object(labels, predictions)
    loss = tf.nn.compute_average_loss(per_example_loss,
                                      global_batch_size=GLOBAL_BATCH_SIZE)
    if model_losses:
      loss += tf.nn.scale_regularization_loss(tf.add_n(model_losses))
    return loss

class CustomModel(tf.keras.Model):
    def __init__(self, model):
        super(CustomModel, self).__init__()
        self.model = model
        self.loss_tracker= tf.keras.metrics.Mean(name='loss')
        
    def train_step(self, data):
        x, y = data
        replica_context = tf.distribute.get_replica_context()
        tf.print("Replica id: ", replica_context.replica_id_in_sync_group, 
                    " of ", replica_context.num_replicas_in_sync)
        GLOBAL_BATCH_SIZE = replica_context.num_replicas_in_sync*tf.shape(y)[0]
        with tf.GradientTape() as tape:
            # Forward pass
            y_pred = self.model(x, training=True)
            # Compute our own loss. Shape (batch_size,)
            loss = tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits = True, reduction=tf.keras.losses.Reduction.NONE)(y, y_pred)
            loss = tf.nn.compute_average_loss(loss, global_batch_size=GLOBAL_BATCH_SIZE)
            
        # Compute gradients
        trainable_vars = self.model.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics
        self.loss_tracker.update_state(loss)
        self.compiled_metrics.update_state(y, y_pred)
        metrics = {m.name: m.result() for m in self.metrics}
        return metrics