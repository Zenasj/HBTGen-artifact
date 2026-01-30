from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

def build_model():
    model = keras.Sequential([
        layers.Dense(24, activation='relu', input_shape=[len(features)]),
        layers.Dense(12, activation='relu'),
        layers.Dense(1)
    ])
    optimizer = tf.keras.optimizers.RMSprop(0.001)
    model.compile(loss='mse',
                    optimizer=optimizer,
                    metrics=['mae', 'mse'])
    return model

model = KerasRegressor(build_model())

pipeline = Pipeline(steps=[ ('nn', model)])

gs = GridSearchCV(estimator=pipeline, param_grid=params,
                  n_jobs=1, return_train_score=True, scoring='r2');

import tensorflow as tf
from tensorflow import keras
import numpy as np


from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool
from functools import partial

def simple_model():
    model = keras.models.Sequential([
        keras.layers.Dense(units = 10, input_shape = [1]),
        keras.layers.Dense(units = 1, activation = 'sigmoid')
    ])
    model.compile(optimizer = 'sgd', loss = 'mean_squared_error')
    return model

def clone_model(model):
    model_clone = tf.keras.models.clone_model(model)
    model_clone.set_weights(model.get_weights())
    return model_clone

def work(model, seq):
    return model.predict(seq)

def worker(model, n = 4):
    seqences = np.arange(0,100).reshape(n, -1)
    pool = Pool()
    # model_list = [clone_model(model) for _ in range(n)]
    # results = pool.map(work, zip(model_list,seqences))
    partial_work = partial(work, model=model)
    results = pool.map(partial_work, seqences)
    pool.close()
    pool.join()
    
    return np.reshape(results, (-1, ))



if __name__ == '__main__':
    model = simple_model()
    out = worker(model, n=4)
    print(out)