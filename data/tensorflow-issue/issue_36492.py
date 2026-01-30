from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf
import tensorflow.keras as keras
import numpy as np

print("TF VERSION: ", tf.__version__)

inputs = keras.Input(2)
d1 = keras.layers.Dense(4)
d2 = keras.layers.Dense(4)
o1 = d1(inputs)
o2 = d2(inputs)

# make a model with multiple outputs
model = keras.Model(inputs=inputs, outputs=[o1, o2])

# compile the model with multiple losses
model.compile(loss=[keras.losses.MeanSquaredError(), keras.losses.MeanSquaredError()])

# try to feed a batch through the model
batch = np.linspace(0, 9, 10).reshape(5, 2)
outs = model.predict(batch)
print(outs)

# save and load the model
model.save("model.hdf5")

model = keras.models.load_model("model.hdf5")

def compile_args_from_training_config(training_config, custom_objects=None):
  """Return model.compile arguments from training config."""
  if custom_objects is None:
    custom_objects = {}

  optimizer_config = training_config['optimizer_config']
  optimizer = optimizers.deserialize(
      optimizer_config, custom_objects=custom_objects)

  # Recover loss functions and metrics.
  loss_config = training_config['loss']  # Deserialize loss class.
  if isinstance(loss_config, dict) and 'class_name' in loss_config:
    loss_config = losses.get(loss_config)
  loss = nest.map_structure(
      lambda obj: custom_objects.get(obj, obj), loss_config)
  metrics = nest.map_structure(
      lambda obj: custom_objects.get(obj, obj), training_config['metrics'])
  weighted_metrics = nest.map_structure(
      lambda obj: custom_objects.get(obj, obj),
      training_config.get('weighted_metrics', None))
  sample_weight_mode = training_config['sample_weight_mode']
  loss_weights = training_config['loss_weights']

  return dict(
      optimizer=optimizer,
      loss=loss,
      metrics=metrics,
      weighted_metrics=weighted_metrics,
      loss_weights=loss_weights,
      sample_weight_mode=sample_weight_mode)