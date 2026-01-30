import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import numpy as np
import keras
import tensorflow as tf
import sklearn.model_selection

X = np.random.randint(low=300, high=900, size=(215699, 30, 30))
y = np.random.randint(low=0, high=51, size=(215699,))
X_trn, X_val, y_trn, y_val = sklearn.model_selection.train_test_split(X, y)

m = tf.keras.Sequential(
    [
        keras.layers.Input(shape=X.shape[1:]),
        keras.layers.Flatten(),
        keras.layers.Dense(units=64, activation="relu",),
        keras.layers.Dense(units=64, activation="relu",),
        keras.layers.Dense(len(np.unique(y))),
    ]
)
m.compile(
    optimizer=keras.optimizers.Adam(1e-5),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
)

sample_weights_val = np.ones(y_val.shape)

history = m.fit(
    X_trn,
    y_trn,
    validation_data=(X_val, y_val, sample_weights_val),
    batch_size=128,
    epochs=5,
)

py
model.fit(
    X_trn, 
    y_trn,
    # NOTE: Validation loss is *incorrect* because it doesn't take into account the class weights
    validation_data=(X_val, y_val),
    class_weight=calculate_weights_for_my_imbalanced_classes(y_trn),
)

py
import numpy as np
import keras
import tensorflow as tf
import sklearn.model_selection

X = np.random.randint(low=300, high=900, size=(215699, 30, 30))
y = np.random.randint(low=0, high=51, size=(215699,))
X_trn, X_val, y_trn, y_val = sklearn.model_selection.train_test_split(X, y)

m = tf.keras.Sequential(
    [
        keras.layers.Input(shape=X.shape[1:]),
        keras.layers.Flatten(),
        keras.layers.Dense(units=64, activation="relu",),
        keras.layers.Dense(units=64, activation="relu",),
        keras.layers.Dense(len(np.unique(y))),
    ]
)
m.compile(
    optimizer=tf.keras.optimizers.legacy.Adam(1e-5),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
)

sample_weights_val = np.ones(y_val.shape)

history = m.fit(
    X_trn,
    y_trn,
    validation_data=(X_val, y_val, sample_weights_val),
    batch_size=128,
    epochs=5,
)