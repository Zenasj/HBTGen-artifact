import random
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf # same issue with tf-nightly = "2.15.0.dev20230904"
import time
import gc
import psutil # psutil == "5.9.5"
import subprocess as sp

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(
        device=gpu, enable=True
    )


def get_cpu_memory():
    memory_info = psutil.virtual_memory()
    # you can have the percentage of used RAM
    memory_percent = 100.0 - memory_info.percent
    memory_free_values = memory_info.available / (1024 * 1024)  # in MB
    # you can calculate percentage of available memory
    return memory_free_values, memory_percent


def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)][0]
    memory_percent = (memory_free_values / 24564) * 100  # my gpu has 24564 MB of memory
    return memory_free_values, memory_percent


class MyModel(tf.keras.Model):

    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(1000, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(10000, activation=tf.nn.relu)
        self.dense3 = tf.keras.layers.Dense(10000, activation=tf.nn.relu)
        self.dense4 = tf.keras.layers.Dense(1000, activation=tf.nn.softmax)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        return x


if __name__ == '__main__':
    print(f"Starting..")
    memory_free_val_initial, memory_perc_initial = get_cpu_memory()
    print(f"[Memory monitoring] Free memory CPU {memory_free_val_initial} MB, {memory_perc_initial} %.")
    memory_free_val_initial_gpu, memory_perc_initial_gpu = get_gpu_memory()
    print(f"[Memory monitoring] Free memory GPU {memory_free_val_initial_gpu} MB, {memory_perc_initial_gpu} %.")

    for r in range(0, 1000):
        model = MyModel()
        # ds = tf.data.Dataset.from_tensor_slices((tf.random.uniform((64*4, 1000)), tf.ones((64*4))))
        ds = (lambda: tf.data.Dataset.from_tensor_slices((tf.random.uniform((64 * 20, 1000)), tf.ones((64 * 20)))))
        model.compile(optimizer='sgd', loss=tf.keras.losses.SparseCategoricalCrossentropy())

        model.fit(ds().batch(64), verbose=0)
        model.evaluate(ds().batch(64), verbose=0)
        tf.keras.backend.clear_session()

        if r % 5 == 0:
            # print every 5 model.fit
            print(f"Round: {r}")
            memory_free_val, memory_perc = get_cpu_memory()
            print(f"[Memory monitoring] Free memory CPU {memory_free_val} MB, {memory_perc} %.")
            memory_free_val_gpu, memory_perc_gpu = get_gpu_memory()
            print(f"[Memory monitoring] Free memory GPU {memory_free_val_gpu} MB, {memory_perc_gpu} %.")
            if r == 0:
                memory_free_first = memory_free_val
                memory_free_first_gpu = memory_free_val_gpu
            # time.sleep(2)

        del model
        gc.collect()
        del ds

    print(f"[Memory monitoring CPU] Memory usage increased by {memory_free_first - memory_free_val} MB, "
          "during the process.")
    print(f"[Memory monitoring GPU] Memory usage increased by {memory_free_first_gpu - memory_free_val_gpu} MB, "
          "during the process.")