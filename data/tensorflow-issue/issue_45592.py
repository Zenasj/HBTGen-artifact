import random
from tensorflow.keras import layers

# built in
import psutil

# third party
import numpy as np
import tensorflow as tf

from dask import compute, delayed
from dask.distributed import Client
from scikeras._utils import make_model_picklable 
from tensorflow.keras import Model
from tensorflow.keras import regularizers
from tensorflow.keras.backend import clear_session
from tensorflow.keras.layers import (
    Dense,
    Input,
    GRU,
    BatchNormalization,
    Dropout,
    Conv1D,
    Flatten
)

def build_keras_model(
        X,
        nodes=15,
        activation="selu",
        kernel_initializer='random_uniform',
        regularizer_l1=0.05,
        regularizer_l2=0.0,
        recurrent_dropout=0.0,
        dropout=0.0,
        dense_units=tuple(),
        batchnorm=True,
        batchnorm_trainable=True,
        batchnorm_training=False,
        use_bias=False,
        loss='mse',
        optimizer='adam'
):
    shape = X.shape[1:]
    inputs = Input(shape=shape, name='inputs')

    x = Dropout(dropout)(inputs) if dropout else inputs

    x = GRU(
        nodes,
        activation=activation,
        recurrent_dropout=recurrent_dropout,
        return_sequences=False,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=regularizers.l1_l2(regularizer_l1, regularizer_l2),
        activity_regularizer=regularizers.l1(0.0),
        use_bias=use_bias
    )(x)

    dense_units = dense_units or ()
    for n in dense_units:
        x = Dense(
            n,
            activation=activation,
            name=f'extra_dense{n}',
            kernel_regularizer=regularizers.l1_l2(0.01, 0.01)
        )(x)

    if batchnorm:
        x = BatchNormalization(trainable=batchnorm_trainable)(x, training=batchnorm_training)

    x = Dense(
        1,
        activation='linear',
        use_bias=use_bias,
        kernel_initializer='random_uniform',
        name='prediction'
    )(x)

    model = Model(inputs=inputs, outputs=x)
    model.compile(
        optimizer=optimizer,
        loss=loss
    )
    make_model_picklable(model)
    return model

def get_memory_usage():
    m_perc = psutil.virtual_memory().percent
    m_used = round(psutil.virtual_memory().used / 10e8, 4)
    return m_used, m_perc


def print_memory_usage(msg=0):
    m_used, m_perc = get_memory_usage()
    print(f'{msg}: memory used is {m_used}, {m_perc}% of total memory')


def fit_keras(model, X, y, **fit_params):
    model.fit(X, y, batch_size=fit_params.pop('batch_size', len(X)), **fit_params)
    return model


def run_keras_single(X, y, n_runs):
    models = []
    for i in range(n_runs):
        print_memory_usage(i)
        model = build_keras_model(X)
        model.fit(X, y, batch_size=len(X), verbose=0)
        models.append(model)
    return models


def run_keras_multi(X, y, n_runs):
    print_memory_usage('before')
    keras_model = build_keras_model(X)
    models = compute(*[delayed(fit_keras)(keras_model, X, y, epochs=20, verbose=0) for i in range(n_runs)])
    print_memory_usage('after')
    return models

rng = np.random.default_rng(7)
n_samples = 500
n_timesteps = 10
n_features = 50
loc = 0
scale = 0.01

X = rng.normal(loc=loc, scale=scale, size=(n_samples, n_timesteps, n_features))
y = rng.normal(loc=loc, scale=scale, size=(n_samples,))

n_runs = 1000

models_single = run_keras_single(X, y, n_runs)
models_multi = run_keras_multi(X, y, n_runs)