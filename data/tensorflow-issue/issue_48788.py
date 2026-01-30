import random

class Data(k.utils.Sequence):
    """
    Converts fit() into fit_generator() interface.
    """

    def __init__(self, inputs, outputs, sample_weights, batch_size, shuffle):
        self._inputs = inputs
        self._outputs = outputs
        self._sample_weights = sample_weights
        self._size = inputs[0].shape[0]
        self._batch_size = batch_size
        self._num_batches = int((self._size-1)/batch_size) + 1
        self._shuffle = shuffle
        self._ids = np.arange(0, self._size)
        self._reshuffle()
        print("\nTotal samples: {} ".format(self._size))
        print("Batch size: {} ".format(min(self._batch_size, self._size)))
        print("Total batches: {} \n".format(self._num_batches))

    def __len__(self):
        return self._num_batches

    def __getitem__(self, index):
        start = index * self._batch_size
        end = min(start + self._batch_size, self._size)
        ids = self._ids[start: end]
        inputs = [v[ids, :] for v in self._inputs]
        outputs = [v[ids, :] for v in self._outputs]
        sample_weights = [v[ids] for v in self._sample_weights]
        return inputs, outputs, sample_weights
    
    def on_epoch_end(self):
        self._reshuffle()

    def get_data(self):
        return self._inputs, self._outputs, self._sample_weights

    def _reshuffle(self):
        if self._num_batches > 1 and self._shuffle:
            self._ids = np.random.choice(self._size, self._size, replace=False)

from tensorflow import keras as k
import numpy as np 

x = k.Input((1,))
l1 = k.layers.Dense(10, activation='tanh')(x)
y = k.layers.Dense(1)(l1)

model = k.Model(x, y)
model.compile(loss=k.losses.MSE)

inputs = [np.linspace(0, 1, 1000).reshape(-1,1)]
outputs = list(map(lambda x: np.sin(2*x), inputs))
weights = list(map(lambda x: np.ones_like(x), inputs))

dg = Data2(inputs, outputs, weights, 32, True)

model.fit(dg, epochs=100)

import sciann as sn 
import numpy as np 
from tensorflow import keras as k

x = sn.Variable('x')
y = sn.Functional('y', x, [10], 'tanh')

model = sn.SciModel(x, y)

inputs = [np.linspace(0, 1, 1000).reshape(-1,1)]
outputs = list(map(lambda x: np.sin(2*x), inputs))
weights = list(map(lambda x: np.ones_like(x).flatten(), inputs))

dg = Data(inputs, outputs, weights, 32, True)

model.train(dg, epochs=100)