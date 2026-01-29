# tf.random.uniform((B, 200, 1), dtype=tf.float32)

import tensorflow as tf
import numpy as np

class CustomGenerator(tf.keras.utils.Sequence):
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
        Shuffle (if applicable) and find exploration order.
        """
        if self.shuffle is True:
            idx_list = np.copy(idx_list)
            np.random.shuffle(idx_list)

        return idx_list

    def _data_generation(self, inputs, labels, idx_list_temp):
        x = inputs[idx_list_temp]
        y = labels[idx_list_temp]
        return x, y

    def __getitem__(self, index):
        # Guard against overshoot - wrap around or reset index if needed
        if self.current_idx + self.batch_size > len(self.idx_list):
            # Reset for safety to avoid empty batches (key problem in issue)
            self.on_epoch_end()
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


class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define layers matching the Flatten-containing model (from issue)
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(units=10)

    def call(self, inputs, training=False):
        # inputs shape expected: (batch, 200, 1)
        x = self.flatten(inputs)
        x = self.dense(x)
        return x


def my_model_function():
    # Return an instance of MyModel
    model = MyModel()
    model.compile(loss='mse', optimizer='adam')
    return model


def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    # According to the issue: input shape (batch, 200, 1)
    B = 64  # batch size
    H = 200
    W = 1
    # Use float32 as typical default dtype in Keras
    return tf.random.uniform((B, H, W), dtype=tf.float32)

