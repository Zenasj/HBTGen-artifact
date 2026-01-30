import random
from tensorflow import keras
from tensorflow.keras import layers

import gc
import time
from itertools import islice
import tracemalloc
from os.path import basename

import tensorflow as tf
import numpy as np

TRACEMALLOC_EVERY = 3000
TRACEMALLOC_IGNORE = "tracemalloc.py:",

input_tensor = tf.keras.Input(shape=(3,), name='input')
output_tensor = tf.keras.layers.Dense(3, name='output')(input_tensor)
model = tf.keras.Model(inputs=[input_tensor], outputs=[output_tensor])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['categorical_accuracy'])


def random_input():
    return np.random.random_sample([1, 3])

model.predict_on_batch(random_input())

tracemalloc.start(50)

last_snapshot = tracemalloc.take_snapshot()
already_printed_from_output_shape_cache = []

while True:
    for _ in range(TRACEMALLOC_EVERY):
        model.predict_on_batch(random_input())

    gc.collect()
    time.sleep(0.1)
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.compare_to(last_snapshot, 'traceback')
    last_snapshot = snapshot

    notable_mem_usage_diff = "\n".join(
        f"{tracemalloc._format_size(stat.size_diff, True):10} "
        f"({tracemalloc._format_size(stat.size_diff // TRACEMALLOC_EVERY, True):7} per prediction) "
        f"{stat.count_diff:+4} objs "
        f"({stat.count_diff / TRACEMALLOC_EVERY:.2f} per prediction) at:\n"
        + "\n".join("    %s" % tb for tb in stat.traceback)
        for stat in islice(
            (s for s in top_stats
             if not any(i in str(tb) for tb in s.traceback for i in TRACEMALLOC_IGNORE)
             and not all(basename(__file__) + ":" in str(tb) for tb in s.traceback)
             and abs(s.size_diff) > 0  # // TRACEMALLOC_EVERY >= 8
             ),
            20
        )
    )
    if notable_mem_usage_diff:
        print("Notable memory usage diff over last %d predictions:\n%s\n" % (TRACEMALLOC_EVERY, notable_mem_usage_diff))
    else:
        print("No notable memory usage diff over last %d predictions\n" % (TRACEMALLOC_EVERY,))
    new_cache_elements = [item for item in model._output_shape_cache.items()
                          if item not in already_printed_from_output_shape_cache]
    if new_cache_elements:
        print("model._output_shape_cache has grown to %d, new elements are:\n%s\n" % (
            len(model._output_shape_cache),
            "\n".join(str(item) for item in new_cache_elements)
        ))
        already_printed_from_output_shape_cache.extend(new_cache_elements)