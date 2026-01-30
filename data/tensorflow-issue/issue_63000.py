import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import os
import numpy as np
import tempfile
import tensorflow as tf
print(tf.__version__)

x = np.random.randn(1, 2, 2, 3)
_input = tf.keras.layers.Input(x.shape[1:])
const = tf.convert_to_tensor(np.ones((1, 1, 1, 3)).astype(np.float32))
_out = tf.keras.layers.Add()([_input, const])
model = tf.keras.Model(inputs=_input, outputs=_out)

_, tmp_h5_file = tempfile.mkstemp('.keras')
tf.keras.models.save_model(model, tmp_h5_file)
loaded_model = tf.keras.models.load_model(tmp_h5_file)  # <== this line fails
os.remove(tmp_h5_file)