from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf

class PrintMixin:
    def custom_print(self):
        print("Hello world")

class CustomModel(tf.keras.models.Model, PrintMixin):
    def __init__(self, *args, **kwargs):
        my_input = tf.keras.layers.Input(shape=(16,))
        dense = tf.keras.layers.Dense(32, activation='relu')
        output = dense(my_input)
        outputs = {"output": output}

        super().__init__(inputs=[my_input], outputs=outputs, *args, **kwargs)


my_model = CustomModel()

def inject_functional_model_class(cls):
  """Inject `Functional` into the hierarchy of this class if needed."""
  from tensorflow.python.keras.engine import functional  # pylint: disable=g-import-not-at-top
  from tensorflow.python.keras.engine import training_v1  # pylint: disable=g-import-not-at-top
  if cls == Model or cls == training_v1.Model:
    return functional.Functional

  cls.__bases__ = tuple(inject_functional_model_class(base)
                        for base in cls.__bases__)
  # Trigger any `__new__` class swapping that needed to happen on `Functional`
  # but did not because functional was not in the class hierarchy.
  cls.__new__(cls)

  return cls

def inject_functional_model_class(cls):
  """Inject `Functional` into the hierarchy of this class if needed."""
  from tensorflow.python.keras.engine import functional  # pylint: disable=g-import-not-at-top
  from tensorflow.python.keras.engine import training_v1  # pylint: disable=g-import-not-at-top
  if cls == Model or cls == training_v1.Model:
    return functional.Functional
  if cls == 'object':
    return cls

  cls.__bases__ = tuple(inject_functional_model_class(base)
                        for base in cls.__bases__)
  # Trigger any `__new__` class swapping that needed to happen on `Functional`
  # but did not because functional was not in the class hierarchy.
  cls.__new__(cls)

  return cls