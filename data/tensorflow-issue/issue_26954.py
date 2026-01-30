from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf

model = tf.keras.Sequential(
    [tf.keras.layers.Dense(5),
     tf.keras.layers.Dense(6),
     tf.keras.layers.Dense(7)])
model(tf.constant([[1.]]))  # create variables

def visit_and_filter(obj, substitute, variable_predicate):
  for k, v in obj._checkpoint_dependencies:
    if isinstance(v, tf.Variable):
      if variable_predicate(v):
        setattr(substitute, k, v)
    elif isinstance(v, tf.keras.layers.Layer):
      v_substitute = tf.keras.Model()
      setattr(substitute, k, v_substitute)
      visit_and_filter(v, v_substitute, variable_predicate)

mirrored = tf.keras.Model()
visit_and_filter(model, mirrored, lambda v: "bias" in v.name)

print([w.name for w in model.weights])
print([w.name for w in mirrored.weights])