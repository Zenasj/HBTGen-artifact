import random
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
from threading import Thread

def build_model():
    layer_1 = tf.keras.layers.Dense(1, input_shape=(20,))
    layer_2 = tf.keras.layers.Activation(tf.sigmoid)
    tf.keras.Sequential([layer_1, layer_2])

for i in range(100):
    t = Thread(target=build_model)
    t.start()

import multiprocessing as mp

def run_as_child_process(f):
    """Decorator which makes the input function run as a child process to the
    thread on which it was called. Requires that the function's return value is
    pickleable (this is a requirement in order to put an object in a
    `multiprocessing.Queue` object) or else it will fail silently.

    Args:
        f: A function whose return value is pickleable.

    Returns:
        A function.
    """

    def subprocess_f(q, *args, **kwargs):
        """Takes in a multiprocessing.Queue object and the positional and
        keyword arguments for `f`. Runs `f` and places the return value in the
        queue for retrieval by the parent process.

        Args:
            q: A multiprocessing.Queue object.
        """
        ret = f(*args, **kwargs)
        q.put(ret)

    def new_f(*args, **kwargs):
        """Spawns a new process in which to run `f`. `q.get()` blocks until it
        receives data from the child process and then returns that data."""
        q = mp.Queue()
        new_args = tuple([q] + list(args))
        p = mp.Process(target=subprocess_f, args=new_args, kwargs=kwargs)
        p.start()
        ret = q.get()
        p.join()
        return ret

    return new_f

3
def build_model():
            with tf.compat.v1.Session() as sess:
                gra = tf.Graph()
                with gra.as_default():
                        layer_1 = tf.keras.layers.Dense(1, input_shape=(20,))
                        layer_2 = tf.keras.layers.Activation(tf.sigmoid)
                        tf.keras.Sequential([layer_1, layer_2])

import tensorflow as tf
from threading import Thread
import multiprocessing as mp
from uuid import uuid4
import time
import numpy as np

def get_sleep_time():
    return abs(np.random.normal(loc=0.1, scale=0.5))

def build_model(i, z):
    time.sleep(get_sleep_time())
    layer_1 = tf.keras.layers.Dense(1, input_shape=(20,))
    layer_2 = tf.keras.layers.Activation(tf.sigmoid)
    model = tf.keras.Sequential([layer_1, layer_2], str(i) + str(uuid4()))
    z.put(model.name)
    return model.name

z = mp.Queue()
for i in range(100):
    t = Thread(target=build_model, args=(i, z))
    t.start()

3
def build_model():
                gra = tf.Graph()  # if this is in a class, you should save it with self.gra
                with gra.as_default():
                        layer_1 = tf.keras.layers.Dense(1, input_shape=(20,))
                        layer_2 = tf.keras.layers.Activation(tf.sigmoid)
                        tf.keras.Sequential([layer_1, layer_2])