from tensorflow.keras import models

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.utils import Sequence


class Dataset(Sequence):
    def __init__(self):
        pass

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return tf.SparseTensor([[0, 1], [1, 0]], [1, 1], (2, 2))


class Net(Model):
    def __init__(self):
        super().__init__()

    def call(self, x, **kwargs):
        return x.values


net = Net()
net.compile()
net.fit(Dataset())