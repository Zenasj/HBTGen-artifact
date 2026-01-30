import random
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import attr
import numpy as np

from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.optimizers import SGD
from tensorflow.python.keras.utils import Sequence


def make_model(D):
    x = Input(shape=[D])
    y = Dense(1, activation=None)(x)
    model = Model(inputs=x, outputs=y)
    model.compile(optimizer=SGD(), loss="mean_squared_error")
    return model


@attr.s
class Feed(Sequence):
    foo = attr.ib(default=np.arange(12))  # Remove this attribute to get the expected behaviour
    batch_size = attr.ib(default=10)
    feature_dimension = attr.ib(default=3)

    def __getitem__(self, idx):
        features = np.random.randn(self.batch_size, self.feature_dimension)
        targets = np.sum(features, axis=1, keepdims=True)
        return features, targets

    def __len__(self):
        return 100


if __name__ == "__main__":
    feed = Feed()
    model = make_model(feed.feature_dimension)
    model.fit(x=feed, epochs=10)

def _get_num_samples_or_steps(data, steps_per_epoch):
  """Returns number of samples or steps, and whether to use steps count mode."""
  flat_inputs = nest.flatten(data)
  if hasattr(flat_inputs[0], 'shape'):
    return int(flat_inputs[0].shape[0]), False
  return steps_per_epoch, True