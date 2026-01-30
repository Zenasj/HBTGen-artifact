import random
from tensorflow.keras import layers
from tensorflow.keras import models

import os
import sys
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation

per_replica_batch_size = 10
num_workers = 2
global_batch_size = per_replica_batch_size * num_workers
dataset_size = 100
steps_per_epoch = 100 / global_batch_size

os.environ['TF_CONFIG'] = json.dumps({
    'cluster': {
        'worker': ["localhost:20000", "localhost:20001"]
    },
    'task': {'type': 'worker', 'index': int(sys.argv[1])}
})
strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

def get_model():
    model = Sequential()
    model.add(Dense(1, input_shape=(1,)))
    model.compile(loss='mse', optimizer='sgd')
    return model

if __name__ == "__main__":
    X_train = np.random.rand(dataset_size, 1)
    noise = np.random.normal(0, 0.01, X_train.shape)
    Y_train = 10 * X_train + 2 + noise
    dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).batch(global_batch_size)

    with strategy.scope():
        model = get_model()
    model.fit(dataset, epochs=1, steps_per_epoch=steps_per_epoch, verbose=2)
    print(model.get_weights())