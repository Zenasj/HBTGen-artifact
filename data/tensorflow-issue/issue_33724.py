@tf.function
def train_one_epoch(model, dataset):
    for x in tqdm(dataset):
        train_step(model, x)

import tensorflow as tf


class Iterable():
    def __init__(self, iterable):
        self.iterable = iterable

    def __iter__(self):
        for obj in self.iterable:
            yield obj

@tf.function
def f(dataset):
    for x in Iterable(dataset):
        print(x)

dataset = tf.data.Dataset.range(5)
f(dataset)