from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import SGD

inputs = Input(shape=(10,))
output = Dense(3, activation="sigmoid")(inputs)

model = Model(
    inputs=inputs, 
    outputs=output
)

model.compile(
    loss='binary_crossentropy',
    optimizer=SGD(lr=1e-3, momentum=0.9), 
    metrics=[tf.keras.metrics.AUC(multi_label=True)]
)