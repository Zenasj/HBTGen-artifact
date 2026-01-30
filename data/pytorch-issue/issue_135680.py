import random

import os
import re
import torch
import numpy as np
os.environ['KERAS_BACKEND']='torch'
import keras

layer = keras.layers.MaxPooling1D(
    pool_size=2,
    strides=1,
    padding="same",
    data_format="channels_first",
    trainable=True,
    dtype="mixed_float16",
    autocast=True,
)

result_dynamic = layer(
    inputs=np.random.rand(*[3, 5, 4]),
)