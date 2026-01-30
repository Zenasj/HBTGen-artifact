from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

# Import TensorFlow
import tensorflow as tf

# Helper libraries
import numpy as np
import os

print(tf.__version__)

fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), _ = fashion_mnist.load_data()

# Adding a dimension to the array -> new shape == (28, 28, 1)
# We are doing this because the first layer in our model is a convolutional
# layer and it requires a 4D input (batch_size, height, width, channels).
# batch_size dimension will be added later on.
train_images = train_images[..., None]

# Getting the images in [0, 1] range.
train_images = train_images / np.float32(255)

# Padding images because ResNet requires a miniaml shape of (32, 32)
padded_train_images = np.concatenate([
    np.zeros((len(train_images), 2, 28, 1)), 
    train_images, 
    np.zeros((len(train_images), 2, 28, 1))
], axis=1)
padded_train_images = np.concatenate([
    np.zeros((len(train_images), 32, 2, 1)), 
    padded_train_images, 
    np.zeros((len(train_images), 32, 2, 1))
], axis=2)

# If the list of devices is not specified in the
# `tf.distribute.MirroredStrategy` constructor, it will be auto-detected.
strategy = tf.distribute.MirroredStrategy()

print ('Number of devices: {}'.format(strategy.num_replicas_in_sync))

BUFFER_SIZE = len(train_images)

BATCH_SIZE_PER_REPLICA = 64
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

EPOCHS = 10

# We keep only the first images, so that the last GPU receives an empty batch
padded_train_images = padded_train_images[:strategy.num_replicas_in_sync-1]
train_labels = train_labels[:strategy.num_replicas_in_sync-1]

train_dataset = tf.data.Dataset.from_tensor_slices((padded_train_images, train_labels)).shuffle(BUFFER_SIZE).batch(GLOBAL_BATCH_SIZE) 
train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)

def create_model():
  inputs = tf.keras.Input((32, 32, 1))
  preprocessed = tf.keras.layers.Conv2D(3, (1, 1))(inputs) # ResNet requires 3 channels
  features = tf.keras.applications.ResNet50V2(
      include_top=False, 
      input_tensor=preprocessed, 
      pooling="avg", weights=None).output
  logits = tf.keras.layers.Dense(10)(features)
  return tf.keras.Model(inputs, features)

with strategy.scope():
  # Set reduction to `none` so we can do the reduction afterwards and divide by
  # global batch size.
  loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
      from_logits=True,
      reduction=tf.keras.losses.Reduction.NONE)
  def compute_loss(labels, predictions):
    per_example_loss = loss_object(labels, predictions)
    return tf.nn.compute_average_loss(per_example_loss, global_batch_size=GLOBAL_BATCH_SIZE)

# model, optimizer, and checkpoint must be created under `strategy.scope`.
with strategy.scope():
  model = create_model()

  optimizer = tf.keras.optimizers.Adam()

  checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)

def train_step(inputs):
  images, labels = inputs

  with tf.GradientTape() as tape:
    predictions = model(images, training=True)
    loss = compute_loss(labels, predictions)

  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  return loss 

# `run` replicates the provided computation and runs it
# with the distributed input.
@tf.function
def distributed_train_step(dataset_inputs):
  per_replica_losses = strategy.run(train_step, args=(dataset_inputs,))
  return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                         axis=None)

for epoch in range(EPOCHS):
  # TRAIN LOOP
  total_loss = 0.0
  num_batches = 0
  for x in train_dist_dataset:
    total_loss += distributed_train_step(x)
    num_batches += 1
  train_loss = total_loss / num_batches

  print(f"Epoch {epoch+1}, Loss: {train_loss}")

def convert_to_sync_batch_norm(old_model: tf.keras.Model, input_layer: tf.keras.Input):
    old_layer_names = [layer.name for layer in old_model.layers]
    new_xs = [input_layer]
    for old_layer in old_model.layers[1:]:
        if isinstance(old_layer.input, list):
            input_x = [new_xs[old_layer_names.index(l.name.split("/")[0])] for l in old_layer.input]
        else:
            input_x = new_xs[old_layer_names.index(old_layer.input.name.split("/")[0])]
        if isinstance(old_layer, tf.keras.layers.BatchNormalization):
            old_layer = tf.keras.layers.experimental.SyncBatchNormalization.from_config(
                old_layer.get_config()
            )
        x = old_layer(input_x)
        new_xs.append(x)

    new_model = tf.keras.Model(new_xs[0], new_xs[-1])
    for old_layer, new_layer in zip(old_model.layers, new_model.layers):
        new_layer.set_weights(old_layer.get_weights())

    return new_model