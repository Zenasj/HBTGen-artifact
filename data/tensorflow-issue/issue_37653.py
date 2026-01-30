import random

# Cleanup utility class
class TfDataset(object):
    def __init__(self):
        self.py_func_set_to_cleanup = set()

    def from_generator(self, generator, output_types, output_shapes=None, args=None):
        if not hasattr(tf.compat.v1.get_default_graph(), '_py_funcs_used_in_graph'):
            tf.compat.v1.get_default_graph()._py_funcs_used_in_graph = []
        py_func_set_before = set(tf.compat.v1.get_default_graph()._py_funcs_used_in_graph)
        result = tf.data.Dataset.from_generator(generator, output_types, output_shapes, args)
        py_func_set_after = set(tf.compat.v1.get_default_graph()._py_funcs_used_in_graph) - py_func_set_before
        self.py_func_set_to_cleanup |= py_func_set_after
        return result
  
    def cleanup(self):
        new_py_funcs = set(tf.compat.v1.get_default_graph()._py_funcs_used_in_graph) - self.py_func_set_to_cleanup
        tf.compat.v1.get_default_graph()._py_funcs_used_in_graph = list(new_py_funcs)
        self.py_func_set_to_cleanup = set()

# Usage example
tf_dataset = TfDataset()
dataset = tf_dataset.from_generator(generator, output_types=tf.int32, output_shapes=[None])
del dataset
tf_dataset.cleanup()  # Call this after done using the generator.

strategy = tf.distribute.MirroredStrategy()
train_iterator = strategy.make_dataset_iterator(train_dataset)
def train_step(inputs):
    images, labels = inputs
    with tf.GradientTape() as tape:
        y_pred = recognizer(images, training=True)
        loss = compute_loss(
                   y_pred, labels
        )
    losses.update_state(loss)
    gradients = tape.gradient(loss, recognizer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, recognizer.trainable_variables))
@tf.function
def distributred_train(dataset):
        return strategy.experimental_run(train_step, dataset)

for epoch in range(epochs):
    for steps in range(train_steps_per_epoch):
        ditributred_train(train_iterator)
        # it use the train_iterator to run the distributed training step.
        # How should I suppose to reset the dataset?

import gc

import psutil
import tensorflow as tf


def generator():
    yield tf.random.randn((1000, 1000))


gc.collect()

process = psutil.Process()

deltas = []

for i in range(1000):
    mem_used_0 = process.memory_info().rss
    # create a dataset from a generator, and cleanup immediatelly
    dataset = tf.data.Dataset.from_generator(
        generator, output_types=tf.float64, output_shapes=[None]
    )
    del dataset
    gc.collect()

    # collect memory usage
    mem_used_1 = process.memory_info().rss

    # store the memory usage delta
    delta = mem_used_1 - mem_used_0
    deltas.append(delta)

# How much memory is leaking?
delta_sum_mb = sum(deltas) / 1024**2
print(f"Sum of memory leaking: {delta_sum_mb:.2f}MB")