import tensorflow as tf
from tensorflow import keras

identifier = "categorical_crossentropy"
tf.keras.losses.get(identifier)

identifier = {"class_name":"categorical_crossentropy","config":{"from_logits":True}}
tf.keras.losses.get(identifier)

def class_and_config_for_serialized_keras_object(
    config,
    module_objects=None,
    custom_objects=None,
    printable_module_name='object'):
  """Returns the class name and config for a serialized keras object."""
  if (not isinstance(config, dict) or 'class_name' not in config or
      'config' not in config):
    raise ValueError('Improper config format: ' + str(config))