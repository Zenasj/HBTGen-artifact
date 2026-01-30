from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

##
##  Imports
##

import sys

import tensorflow as tf

from tensorflow.keras.models     import Model
from tensorflow.keras.layers     import Input, Dense
from tensorflow.keras.optimizers import AdamW

##
##  Report versions
##

print(f"Python version is: {sys.version}")
##  -->  Python version is: 3.10.11 | packaged by conda-forge | (main, May 10 2023, 19:01:19) [Clang 14.0.6 ]

print(f"TF version is: {tf.__version__}")
##  -->  TF version is: 2.12.0

print(f"Keras version is: {tf.keras.__version__}")
##  -->  Keras version is: 2.12.0


##
##  Create a very simple model
##

x_in  = Input(1)
x     = Dense(10)(x_in)
model = Model(x_in, x)

##
##  Compile model with AdamW optimizer
##
model.compile(optimizer=AdamW(learning_rate=1e-3, weight_decay=1e-2))