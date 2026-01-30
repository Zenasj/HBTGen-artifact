import random
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models

import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.layers import Dense, Input
from tensorflow.python.keras.models import Model, Sequential

def build_std_func_model(n_inputs):
    inp = Input(n_inputs)
    h = Dense(100, activation="relu")(inp)
    h = Dense(100, activation="relu")(h)
    h = Dense(100, activation="relu")(h)
    h = Dense(100, activation="relu")(h)
    output = Dense(n_inputs, activation="linear")(h)
    return Model(inputs=inp, outputs=output)

def build_std_seq_model(n_inputs):
    model = Sequential()

    model.add(Input(n_inputs))
    model.add(Dense(100, activation="relu"))
    model.add(Dense(100, activation="relu"))
    model.add(Dense(100, activation="relu"))
    model.add(Dense(100, activation="relu"))
    model.add(Dense(n_inputs, activation="linear"))
    return model

def make_func_dataset(n_examples=30000, n_linears=15, n_categorical=4):
    linear = np.random.randn(n_examples, n_linears)
    categorical = [np.random.choice([0, 1, 2], size=n_examples) for _ in range(n_categorical)]
    categorical = [to_categorical(item) for item in categorical]
    return [linear, *categorical]

def run_simplest():
    X = make_func_dataset(n_examples=10**5, n_linears=30, n_categorical=0)[0]
    # model = build_std_func_model(30)
    model = build_std_seq_model(30)
    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(X, X, epochs=100)

if __name__ == '__main__':
    # tf.config.set_visible_devices([], 'GPU')
    # run_functional()
    run_simplest()