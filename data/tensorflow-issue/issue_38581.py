import random
from tensorflow.keras import layers
from tensorflow.keras import models

import sys
import resource  # used for monitoring memory usage
import logging
import numpy as np

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import callbacks

"""In order to prevent the memory leak remove/comment out the tensorflow 
imports above and uncomment keras imports below"""
# import keras
# from keras.models import Sequential
# from keras.layers import Dense
# from keras import callbacks


class MemoryLoggerCallback(callbacks.Callback):
    def __init__(self):
        self.memory_usage = []

    def on_epoch_end(self, epoch, logs=None):
        max_used_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        logging.info(f"\nmemory usage after end of epoch hook: {max_used_memory}")
        if epoch % 10 == 0:
            self.memory_usage.append(max_used_memory)
            logging.info("############")
            logging.info(self.memory_usage)
            logging.info("############")


def dummy_gen_simple(input_dim: int, batch_size: int = 3):
    """random data and label generator"""
    while True:
        x = np.random.random((batch_size, input_dim))
        y = keras.utils.to_categorical(np.random.randint(10, size=(batch_size, 1)), num_classes=10)
        yield (x, y, None)


def init_logger():
    """initialise logger which redirects to stderr,
    so it prints log messages during training"""
    # set up global logger
    logger = logging.getLogger('')
    logger.setLevel(logging.INFO)
    logger.handlers = []  # remove default handlers
    # set up STDERR handler
    stderr_handler = logging.StreamHandler(sys.stderr)
    logger.addHandler(stderr_handler)


init_logger()

input_dim = 500000

# arbitrary simple model
model = Sequential()
model.add(Dense(64, activation='relu', input_dim=input_dim))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

memory_logger_callback = MemoryLoggerCallback()

model.fit(dummy_gen_simple(input_dim, batch_size=32),
          steps_per_epoch=10,
          epochs=252,
          validation_data=dummy_gen_simple(input_dim, batch_size=16),
          validation_steps=2,
          callbacks=[memory_logger_callback])