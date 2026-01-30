import random

3
import numpy as np

import tensorflow as tf
import tensorflow.keras as tfk
Sequence = tfk.utils.Sequence

Dense = tfk.layers.Dense
Input = tfk.layers.Input
Flatten = tfk.layers.Flatten
Model = tfk.models.Model


class CustomGenerator(Sequence):
    def __init__(self, batch_size, shuffle, steps_per_epoch, data):
        self.inputs = data[0]
        self.labels = data[1]
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch

        # initial idx
        self.idx_list = self._get_exploration_order(range(self.inputs.shape[0]))
        self.current_idx = 0

    def __len__(self):
        return self.steps_per_epoch

    def _get_exploration_order(self, idx_list):
        """
        :param idx_list:
        :return:
        """
        # shuffle (if applicable) and find exploration order
        if self.shuffle is True:
            idx_list = np.copy(idx_list)
            np.random.shuffle(idx_list)

        return idx_list

    def _data_generation(self, inputs, labels, idx_list_temp):
        x = inputs[idx_list_temp]
        y = labels[idx_list_temp]
        return x, y

    def __getitem__(self, index):
        x, y = self._data_generation(self.inputs,
                                     self.labels,
                                     self.idx_list[self.current_idx:self.current_idx + self.batch_size])
        self.current_idx += self.batch_size
        return x, y

    def on_epoch_end(self):
        # shuffle the list when epoch ends for the next epoch
        self.idx_list = self._get_exploration_order(range(self.inputs.shape[0]))
        # reset counter
        self.current_idx = 0

# Model 1 which does not have Flatten
input_tensor = Input(shape=[200], name='input')
output_tensor = Dense(units=10, name='output')(input_tensor)
neuralnet = Model(inputs=input_tensor, outputs=output_tensor)
neuralnet.compile(loss='mse', optimizer='adam')

# Model 2 which has Flatten
input_tensor = Input(shape=[200, 1], name='input')
flat = Flatten()(input_tensor)
output_tensor = Dense(units=10, name='output')(flat)
neuralnet_flat = Model(inputs=input_tensor, outputs=output_tensor)
neuralnet_flat.compile(loss='mse', optimizer='adam')

3
predgen = CustomGenerator(batch_size=64, shuffle=True, steps_per_epoch=10, data=[np.random.normal(size=(700, 200, 1)), np.random.normal(size=(700, 10))])

neuralnet_flat.fit_generator(generator=predgen, epochs=5)

3
predgen = CustomGenerator(batch_size=64, shuffle=True, steps_per_epoch=10, data=[np.random.normal(size=(700, 200)), np.random.normal(size=(700, 10))])
valgen = CustomGenerator(batch_size=64, shuffle=True, steps_per_epoch=1, data=[np.random.normal(size=(64, 200)), np.random.normal(size=(64, 10))])

neuralnet.fit_generator(generator=predgen, validation_data=valgen, epochs=5)

3
predgen = CustomGenerator(batch_size=64, shuffle=True, steps_per_epoch=10, data=[np.random.normal(size=(700, 200, 1)), np.random.normal(size=(700, 10))])
valgen = CustomGenerator(batch_size=64, shuffle=True, steps_per_epoch=1, data=[np.random.normal(size=(64, 200, 1)), np.random.normal(size=(64, 10))])

# does not work
neuralnet_flat.fit_generator(generator=predgen, validation_data=valgen, epochs=5)