"""Testing tensorflow.keras autocompletion."""
# pyright: basic, reportMissingTypeStubs=false

''' Test 1: tf.keras.<autocompletion>'''
import tensorflow as tf

tf.keras.Model()
tf.keras.preprocessing.image_dataset_from_directory(None)

# Broken in 2.5.0, 2.7.0, 2.9.0 (resolves to (keras: LazyLoader) -> everything resolves to Any == No autocompletion)
# Works in 2.10.0-dev20220531

''' Test 2: from tf import keras -> keras.<autocompletion>'''
from tensorflow import keras

keras.Model()
keras.preprocessing.image_dataset_from_directory(None)

# Broken in 2.5.0, 2.7.0, 2.9.0 (resolves to (keras: LazyLoader) -> everything resolves to Any == No autocompletion)
# Broken in 2.10.0-dev20220531 (keras is resolved to tensorflow/__init__.py)

''' Test 3: from tensorflow.keras import Model'''
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense

# Works in 2.5.0, 2.7.0
# Broken in 2.9.0, 2.10.0-dev20220531 (Import "tensorflow.keras" could not be resolved -> Model: Unknown)

from tensorflow.keras import layers