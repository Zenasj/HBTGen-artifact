import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import os
from typing import Any, Callable, Dict, Text

import numpy as np
import psutil
import tensorflow as tf
from keras.callbacks import TensorBoard
from keras.optimizers import Optimizer
from tensorflow.python.data.ops.options import AutoShardPolicy

NUM_TRAIN_SAMPLES = 1000
NUM_DEV_SAMPLES = 10


def get_curriculum_fn(optimizer: Optimizer):
    """Method to load the curriculum function based on the optimizer iterations.
    :param optimizer: Instance of tf.keras.optimizers.Optimizer.
    :return:
    """

    def curriculum_fn(example):
        """Creates a function returning True or False depending on the step.

        This function can directly be applied as a predicate in a
        `tf.data.Dataset.filter` method.

        :param example: Dictionary of feature tensors.
        :return: True or False
        """
        # Get constants based on train_step
        step = tf.cast(optimizer.iterations, tf.int64)
        max_score = 0.4
        min_score = 0.0
        step_update = 500
        half_life = 1000

        # curriculum_step
        curriculum_step = tf.floor(step / tf.cast(step_update, tf.int64))

        # Get the min score
        delta = max_score - min_score
        weight = 0.5 ** (
                tf.cast(curriculum_step, tf.float32) / tf.cast(half_life, tf.float32)
        )
        min_score = tf.cast(max_score - delta * weight, tf.float32)
        return example["score"][0] >= min_score

    return curriculum_fn


def get_dataset(split: str, batch_size: int, max_length: int = 64, curriculum_fn: Callable = None):
    # Number generator
    def generator():
        for i in range(num_samples):
            _dims = np.random.randint(low=1, high=max_length, size=1)
            x = np.zeros(_dims, dtype=np.int32) + np.random.randint(low=1, high=5000, size=1)
            y = np.zeros(_dims, dtype=np.int32) + np.random.randint(low=1, high=5000, size=1)
            score = np.random.uniform(low=0, high=1, size=1)
            yield {'sources': x, 'targets': y, 'score': score}

    assert split in ("train", "dev")
    is_training = split == "train"

    num_samples = NUM_TRAIN_SAMPLES if is_training else NUM_DEV_SAMPLES
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = AutoShardPolicy.DATA
    dataset = tf.data.Dataset.from_generator(
        lambda: generator(),
        output_signature={
            'sources': tf.TensorSpec(shape=(None,), dtype=tf.int32),
            'targets': tf.TensorSpec(shape=(None,), dtype=tf.int32),
            'score': tf.TensorSpec(shape=(1,), dtype=tf.float32)
        }
    ).with_options(options=options)

    if is_training:
        dataset = dataset.shuffle(buffer_size=256).repeat()

    # Curriculum Learning
    if curriculum_fn is not None and is_training:
        dataset = dataset.filter(curriculum_fn)

    dataset = dataset.padded_batch(batch_size=batch_size)

    def map_to_example(example):
        sources, targets = example['sources'], example['targets']
        return sources, targets

    dataset = dataset.map(map_to_example, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.prefetch(tf.data.AUTOTUNE)


def memory_usage_process(pid=None) -> float:
    """Returns the memory used by a specific process
    :param pid: Process id. If omitted, the default is determined by os.getpid().
    :return: the memory used by the current process in MB.
    """
    # return the memory usage in MB
    import psutil

    process = psutil.Process(pid=pid)
    mem = process.memory_full_info().uss / float(2 ** 20)
    return mem


def get_memory_info(logs: Dict[Text, Any] = None) -> Dict[Text, Any]:
    """Returns the memory usage of RAM and GPU memory in MB.
    :param logs: (Optional). If provided, all entries will be appended to the
       existing log. Otherwise, a new log will be created.
    :return: A dictionary containing RAM and GPU memory usage.
    """
    if logs is None:
        logs = {}

    # Get RAM usage
    logs["virtual_memory/used_system [MB]"] = psutil.virtual_memory().used / float(2 ** 20)
    logs["virtual_memory/used_process"] = memory_usage_process(pid=os.getpid())

    # Get GPU memory usage
    physical_devices = tf.config.list_physical_devices("GPU")
    if len(physical_devices) == 0:
        return logs

    for n, device in enumerate(physical_devices):
        device_name = "{}:{}".format(device.device_type, n)
        logs["gpu_memory/{device_name} [MB]".format(device_name=device_name)] = tf.config.experimental.get_memory_info(
            device_name
        )["current"]
    return logs


class MemoryProfilingCallback(tf.keras.callbacks.Callback):
    def __init__(self, tb: TensorBoard, update_freq: int = 100):
        super().__init__()
        self.tb = tb
        self.update_freq = update_freq
        self.batch_time = 0.0

    def on_train_batch_end(self, batch, logs=None):

        # To account for gradient accumulation we ask the optimizer how many gradient
        # updates happened instead of taking the batch-index
        step = int(self.model.optimizer.iterations.value()) - 1

        if step % self.update_freq == 0:
            metrics = {}
            metrics = get_memory_info(logs=metrics)

            # Write logs
            self._write(metrics, step)

    def _write(self, metrics: dict, step):
        # noinspection PyProtectedMember
        with self.tb._train_writer.as_default():
            for name, value in metrics.items():
                tf.summary.scalar(name, value, step=step)


def get_callbacks(log_dir: str):
    _callbacks = []

    # Tensorboard
    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=os.path.join(log_dir, "logs"))
    _callbacks.append(tensorboard_cb)

    # Memory profiler
    _callbacks.append(MemoryProfilingCallback(tensorboard_cb))

    return _callbacks


def main():
    LOG_DIR = "/tmp/memory_leak_curriculum/"

    max_length = 64

    train_batch_size = 64
    valid_batch_size = 16

    optimizer = tf.keras.optimizers.Adam()

    # Model
    inputs = tf.keras.layers.Input(shape=(None,), dtype=tf.int32)
    x = inputs
    x = tf.keras.layers.Embedding(input_dim=5000,
                                  output_dim=64)(x)
    x = tf.keras.layers.Dense(5000)(x)
    model = tf.keras.Model(inputs=inputs, outputs=x)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer)
    model.summary()

    train_data = get_dataset(
        split="train",
        batch_size=train_batch_size,
        max_length=max_length,
        curriculum_fn=get_curriculum_fn(optimizer=optimizer)
    )

    valid_data = get_dataset(
        split="dev",
        batch_size=valid_batch_size,
        max_length=max_length,
    )

    model.fit(
        train_data,
        epochs=100,
        steps_per_epoch=5000,
        validation_data=valid_data,
        validation_steps=3,
        callbacks=get_callbacks(log_dir=LOG_DIR)
    )


if __name__ == '__main__':
    main()