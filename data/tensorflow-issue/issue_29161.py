from tensorflow.keras import layers

import numpy as np
from tensorflow import keras

INPUT_SIZE = 3
DENSE_OUTPUTS = 2
NUM_OF_SAMPLES = 1000
BATCH_SIZE = 2
NUM_OF_BATCHES = 5


class DummySequence(keras.utils.Sequence):

    def __len__(self):
        return NUM_OF_SAMPLES // BATCH_SIZE

    def __getitem__(self, index):
        data = [np.full(shape=(INPUT_SIZE,), fill_value=(index*BATCH_SIZE + i)) for i in range(BATCH_SIZE)]
        labels = [np.full(shape=(DENSE_OUTPUTS,), fill_value=(index*BATCH_SIZE + i))*INPUT_SIZE for i in range(BATCH_SIZE)]
        return np.stack(data), np.stack(labels)



x = keras.layers.Input(shape=(INPUT_SIZE,))
dense_layer = keras.layers.Dense(DENSE_OUTPUTS)
y = dense_layer(x)
model = keras.Model(x, y)

# remove comment in tf 1.12
#model.compile(optimizer="sgd", loss=keras.losses.mean_squared_error)

shapes = [v.shape for v in dense_layer.weights]
dense_layer.set_weights([np.full(shape=shapes[0], fill_value=1.0), np.full(shape=shapes[1], fill_value=0.0)])

seq = DummySequence()

workers = 5
multiprocessing = True
# works with multi-threaing
#multiprocessing = False
print("running predict with multiprocessing: {}".format(multiprocessing))
res = model.predict(seq, workers=workers, use_multiprocessing=multiprocessing, steps=NUM_OF_BATCHES)
print("predict # of results: {}\nresults:\n{}".format(len(res), res))

from tensorflow.python.keras import backend as K
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
K.set_session(session)

import numpy as np
from tensorflow import keras
#import keras

INPUT_SIZE = 3
DENSE_OUTPUTS = 2
NUM_OF_SAMPLES = 1000
BATCH_SIZE = 2
NUM_OF_BATCHES = 5


class DummySequence(keras.utils.Sequence):

    def __len__(self):
        return NUM_OF_SAMPLES // BATCH_SIZE

    def __getitem__(self, index):
        data = [np.full(shape=(INPUT_SIZE,), fill_value=(index*BATCH_SIZE + i)) for i in range(BATCH_SIZE)]
        labels = [np.full(shape=(DENSE_OUTPUTS,), fill_value=(index*BATCH_SIZE + i))*INPUT_SIZE for i in range(BATCH_SIZE)]
        return np.stack(data), np.stack(labels)


class CountBatchesCallback(keras.callbacks.Callback):

    def __init__(self):
        super(CountBatchesCallback, self).__init__()

        self.batches = 0

    def on_batch_begin(self, batch, logs=None):
        self.batches += 1


def get_model():
    x = keras.layers.Input(shape=(INPUT_SIZE,))
    dense_layer = keras.layers.Dense(DENSE_OUTPUTS)
    y = dense_layer(x)
    model = keras.Model(x, y)
    model.compile(optimizer="sgd", loss=keras.losses.mean_squared_error)
    shapes = [v.shape for v in dense_layer.weights]
    dense_layer.set_weights([np.full(shape=shapes[0], fill_value=1.0), np.full(shape=shapes[1], fill_value=0.0)])
    return model


def run_fit_and_predict(model):
    seq = DummySequence()
    steps = 5
    batch_counter_callback = CountBatchesCallback()
    use_multiprocessing = True
    workers = 5
    print("running fit with {} steps".format(steps))
    model.fit_generator(
        seq,
        epochs=1,
        steps_per_epoch=steps,
        use_multiprocessing=use_multiprocessing,
        # workers=workers,
        callbacks=[batch_counter_callback]
    )
    print("batches processed: {}".format(batch_counter_callback.batches))
    results = model.predict_generator(
        seq,
        use_multiprocessing=use_multiprocessing,
        # workers=workers,
        steps=steps
    )
    print("\npredict\nexpected number of results: {}.\nactual number of results: {}.\npredictions:\n{}".format(
        steps * BATCH_SIZE, len(results), results)
    )


if __name__ == '__main__':
    model = get_model()
    print("\n************************ Running run_fit_and_predict first ************************")
    run_fit_and_predict(model)
    print("\n************************ clear session ************************")
    keras.backend.clear_session()
    model = get_model()
    print("\n************************ Running run_fit_and_predict second ************************")
    run_fit_and_predict(model)

import multiprocess.context as ctx
ctx._force_start_method('spawn')

import logging
from multiprocessing.util import log_to_stderr

log_to_stderr(level=logging.DEBUG)