from tensorflow.keras import layers

import os
import sys
import io
from contextlib import contextmanager
from absl import logging as absl_logging

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

absl_logging.set_verbosity(absl_logging.ERROR)

import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import (
    convert_variables_to_constants_v2,
)
from keras import Sequential
from keras.layers import Dense, Input


# Define a context manager to suppress both stdout and stderr
@contextmanager
def suppress_output():
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    # Create temporary string buffers for both
    temp_stdout = io.StringIO()
    temp_stderr = io.StringIO()
    try:
        # Redirect stdout and stderr to the temporary buffers
        sys.stdout = temp_stdout
        sys.stderr = temp_stderr
        yield
    finally:
        # Restore the original stdout and stderr
        sys.stdout = old_stdout
        sys.stderr = old_stderr


# Define the model
input_shape = (1,)
keras_model = Sequential([Input(shape=input_shape), Dense(1)])
full_model = tf.function(lambda x: keras_model(x))
full_model = full_model.get_concrete_function(
    x=tf.TensorSpec(shape=input_shape, dtype=tf.float32)
)

# Even this suppression does not work
with suppress_output():
    frozen_func = convert_variables_to_constants_v2(
        full_model, lower_control_flow=False
    )
    # Output to STDERR here:
    # WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
    # I0000 00:00:1730280536.054103    6179 devices.cc:67] Number of eligible GPUs (core count >= 8, compute capability >= 0.0): 0

print(f"TensorFlow Version: {tf.__version__}")