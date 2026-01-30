import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import os
import tensorflow as tf
import numpy as np

# Disable logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.logging.set_verbosity(tf.logging.ERROR)
tf.enable_eager_execution()

# Create model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(256, 3, padding="same"),
    tf.keras.layers.Conv2D(3, 3, padding="same")
])
print("Are weights empty before training?", model.weights == [])

# Create optim, checkpoint
optimizer = tf.train.AdamOptimizer(0.001)
checkpoint = tf.train.Checkpoint(model=model)

# Make fake data
img = np.random.uniform(0, 255, (32, 32, 3)).astype(np.float32)
truth = np.random.uniform(0, 255, (32, 32, 3)).astype(np.float32)
# Train
with tf.GradientTape() as tape:
    logits = model(img[None])
    loss = tf.losses.mean_squared_error(truth[None], logits)

# Compute/apply gradients
grads = tape.gradient(loss, model.trainable_weights)
grads_and_vars = zip(grads, model.trainable_weights)
optimizer.apply_gradients(grads_and_vars)

# Save model
checkpoint_path = './ckpt/'
checkpoint.save('./ckpt/')

# Check if weights update
print("Are weights empty after training?", model.weights == [])

# Reset model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(256, 3, padding="same"),
    tf.keras.layers.Conv2D(3, 3, padding="same")
])
print("Are weights empty when resetting model?", model.weights == [])

# Update checkpoint pointer
checkpoint = tf.train.Checkpoint(model=model)
# Restore values from the checkpoint
status = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_path))

print("Are weights empty after restoring from checkpoint?", model.weights == [])
print(status)
status.assert_existing_objects_matched()
status.assert_consumed()

import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(256, 3, padding="same"),
    tf.keras.layers.Conv2D(3, 3, padding="same")
])
print(model.variables)

import numpy as np
import tensorflow as tf

model_1 = tf.keras.Sequential([
    tf.keras.layers.Conv2D(256, 3, padding="same"),
    tf.keras.layers.Conv2D(3, 3, padding="same")
])

model_1(np.random.uniform(0, 255, (1, 32, 32, 3)))
save_path = tf.train.Checkpoint(model=model_1).save("/tmp/tf_ckpts/")

model_2 = tf.keras.Sequential([
    tf.keras.layers.Conv2D(256, 3, padding="same"),
    tf.keras.layers.Conv2D(3, 3, padding="same")
])

restore_checkpoint = tf.train.Checkpoint(model=model_2)
status = restore_checkpoint.restore(save_path)
#status.assert_consumed()  # Fails! model_2.variables is empty
model_2(np.random.uniform(0, 255, (1, 32, 32, 3)))
status.assert_consumed()  # Passes

import tensorflow as tf
import numpy as np

def recreate_model_and_checkpoint():
  model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(256, 3, padding="same"),
    tf.keras.layers.Conv2D(3, 3, padding="same")
  ])
  return tf.train.Checkpoint(
      optimizer=tf.keras.optimizers.Adam(0.1),
      model=model)

def train_step(checkpoint):
  model = checkpoint.model
  optimizer = checkpoint.optimizer
  with tf.GradientTape() as tape:
    output = model(tf.ones([1, 32, 32, 3]))
    loss = tf.reduce_sum(output)
  variables = model.trainable_variables
  gradients = tape.gradient(loss, variables)
  optimizer_weights = []
  before_train_step_weights = [v.numpy() for v in variables]
  optimizer.apply_gradients(zip(gradients, variables))
  return loss, before_train_step_weights

checkpoint_one = recreate_model_and_checkpoint()
# Just to create the variables so we have something to save
train_step(checkpoint_one)
save_path = checkpoint_one.save("/tmp/tf_ckpts/")
original_loss_1, original_variable_values_1 = train_step(checkpoint_one)
original_loss_2, original_variable_values_2 = train_step(checkpoint_one)

checkpoint_two = recreate_model_and_checkpoint()
status = checkpoint_two.restore(save_path)
new_loss_1, new_variable_values_1 = train_step(checkpoint_two)
status.assert_consumed()
new_loss_2, new_variable_values_2 = train_step(checkpoint_two)

np.testing.assert_allclose(new_loss_1.numpy(), original_loss_1.numpy())
np.testing.assert_allclose(new_loss_2.numpy(), original_loss_2.numpy())
assert len(original_variable_values_1) == len(new_variable_values_1)
for original_value, new_value in zip(original_variable_values_1, new_variable_values_1):
  np.testing.assert_allclose(original_value, new_value)
for original_value, new_value in zip(original_variable_values_2, new_variable_values_2):
  np.testing.assert_allclose(original_value, new_value)