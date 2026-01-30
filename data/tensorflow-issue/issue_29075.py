from tensorflow import keras
from tensorflow.keras import layers

import time
import itertools
import random

import tensorflow as tf

def generate_token_based_shapes(num_tokens=4096):
    while True:
        length = random.randint(1, 100)
        batch_size = int(num_tokens / length)
        yield (batch_size, length)

# Generate 500k tensors of shape [None, None, 512] but with similar total size.
shapes = list(itertools.islice(generate_token_based_shapes(), 500000))
dataset = tf.data.Dataset.from_tensor_slices(shapes)
dataset = dataset.shuffle(len(shapes))
dataset = dataset.map(lambda shape: tf.zeros(tf.concat([shape, [512]], axis=0)))
dataset = dataset.repeat()
dataset = dataset.prefetch(1)

# Define a model with some layers.
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1024),
    tf.keras.layers.Dense(1024),
    tf.keras.layers.Dense(1024),
    tf.keras.layers.Dense(1024),
    tf.keras.layers.Dense(1024)])

@tf.function(input_signature=(tf.TensorSpec([None, None, 512], dtype=tf.float32),))
def run_step(inputs):
    return model(inputs)

seen_lengths = set()
for x in dataset:
    length = x.shape[1]
    start = time.time()
    _ = run_step(x)
    end = time.time()
    print(length in seen_lengths, end - start)
    seen_lengths.add(length)

import time
import itertools
import random

import tensorflow as tf

def generate_token_based_shapes(num_tokens=4096):
    while True:
        length = random.randint(1, 100)
        batch_size = int(num_tokens / length)
        yield (batch_size, length)

# Generate 500k tensors of shape [None, None, 512] but with similar total size.
shapes = list(itertools.islice(generate_token_based_shapes(), 500000))
dataset = tf.data.Dataset.from_tensor_slices(shapes)
dataset = dataset.shuffle(len(shapes))
dataset = dataset.map(lambda shape: tf.zeros(tf.concat([shape, [512]], axis=0)))
dataset = dataset.repeat()
dataset = dataset.prefetch(1)
iterator = iter(dataset)

# Define a model with some layers.
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1024),
    tf.keras.layers.Dense(1024),
    tf.keras.layers.Dense(1024),
    tf.keras.layers.Dense(1024),
    tf.keras.layers.Dense(1024)])

@tf.function
def run_step():
    inputs = next(iterator)
    return model(inputs)

seen_lengths = set()
while True:
    start = time.time()
    x = run_step()
    end = time.time()
    length = x.shape[1]
    print(length in seen_lengths, end - start)
    seen_lengths.add(length)