import random

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model

## Using Tensorflow as Keras backend.
## Input dtype default is float32

kwargs={'units': 0}
input = (10 * np.random.random((1,32,32,16)))
layer = layers.Dense(**kwargs)
x = keras.Input(batch_shape=input.shape)
y = layer(x)
bk_model = Model(x, y)
print('finish')