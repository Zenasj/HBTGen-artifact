from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

import numpy as np
import pandas as pd
import random
import tensorflow as tf
import time as tm

INPUT_SHAPE=[7, 9]
NUM_POINTS=2000
BATCH_SIZE=32
BUFFER_SIZE=32
EPOCHS=2

def gen():
    for i in range(1000):
        x = np.random.rand(INPUT_SHAPE[0],INPUT_SHAPE[1])
        y = np.random.randint(1, 3, size=1)
        yield x,y
    
dataset = tf.data.Dataset.from_generator(gen, 
                                         (tf.float32, tf.int16),
                                         output_shapes=(tf.TensorShape(INPUT_SHAPE), tf.TensorShape(None))
                                        ).batch(BATCH_SIZE)

def create_model(input_shape):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(100, activation="tanh", input_shape=input_shape),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(3, activation="softmax")
    ])
    return model

model = create_model(input_shape=INPUT_SHAPE)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4, clipvalue=1.0),
    loss= tf.keras.losses.CategoricalCrossentropy(),
    )

iterator = dataset.make_one_shot_iterator()
x,y = iterator.get_next()
print("X.shape", x.shape, y.shape)
print("Predicted", model.predict(x).shape)

print(model.summary())
model.fit_generator(dataset, epochs=EPOCHS, verbose=2)
model.fit(dataset, epochs=EPOCHS, verbose=2)

import tensorflow as tf               # 2.0.0
import tensorflow_datasets as tfds    # 1.3.0

# Dummy data.
text = [
  "the quick",
  "brown fox",
  "jups over",
  "the lazy",
  "dog."
]
target = [
  "class_1,class_2",
  "class_1",
  "class_2",
  "class_1",
  "class_1,class_2"
]

batch_size = 2

# Make dataset.
dataset = tf.data.Dataset.from_tensor_slices((text, target))

# Text (features) encoder.
corpus_generator = (t[0].numpy() for t in dataset) 
encoder = tfds.features.text.SubwordTextEncoder.build_from_corpus(
  corpus_generator, 
  target_vocab_size=500
)

# Targets lookup.
targets = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(
  keys=["class_1", "class_2"], 
  values=[1,2], 
  value_dtype=tf.int64), default_value=-1
)

vocab_size = encoder.vocab_size
n_targets = targets.size().numpy()
print(f"vocab_size: {vocab_size}\tn_targets: {n_targets}")

# Map fn.
encode = lambda x: [encoder.encode(x.numpy())]
tf_encode = lambda x: tf.reduce_sum(
  tf.one_hot(
    tf.py_function(encode, [x], tf.int64), 
    depth=vocab_size, 
    axis=None, 
    dtype=tf.int64), 
  axis=0, 
  keepdims=False
)

lookup_targets = lambda x: tf.reduce_sum(
  tf.one_hot(
    targets.lookup(tf.strings.split(x, sep=",")),
    depth=n_targets,
    axis=None,
    dtype=tf.float32), 
  axis=0, 
  keepdims=False
)

parse_fn = lambda x, y: (tf_encode(x), lookup_targets(y))
dataset = dataset.map(parse_fn)
dataset = dataset.batch(batch_size)

for batch_x, batch_y in dataset:
  print("*** BATCH ***")
  for i, (x, y) in enumerate(zip(batch_x, batch_y)):
    print(f"> EXAMPLE {i + 1}")
    print(f"text   : {x.numpy()} => shape: {x.shape}")
    print(f"target : {y.numpy()} => shape:{y.shape}")
    print()
  print()

inputs = tf.keras.layers.Input(shape=(vocab_size,))
outputs = tf.keras.layers.Dense(units=n_targets)(inputs)  # no activation!

model = tf.keras.Model(
  inputs=[inputs], 
  outputs=[outputs], 
)

print(f"Sample prediction: {model.predict(dataset.take(1))}\n")

loss = tf.keras.losses.BinaryCrossentropy()
optimizer = tf.keras.optimizers.Adam()

model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

history = model.fit(
  dataset,
  epochs=5,
)

def _fixup_shape(images, labels, weights):
    images.set_shape([None, None, None, 3])
    labels.set_shape([None, 19]) # I have 19 classes
    weights.set_shape([None])
    return images, labels, weights
dataset = dataset.map(_fixup_shape)

Dataset.from_generator(
    gen,
    output_signature=(
        tf.TensorSpec(shape=(None, 32, 32, 3), dtype=tf.float32),
        tf.TensorSpec(shape=None, dtype=tf.float32)
    )
)

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

inp = Input(shape=(3,))
output = Dense(1, activation='sigmoid')(inp)
model = Model(inp, output)
model.compile(optimizer=Adam(1e-2),
              loss='binary_crossentropy',
)

def simple_generator():
    while True:
        yield [0.5, 0.2, -0.3], 0.0
        yield [-0.5, 0.3, -0.1], 1.0

dataset = tf.data.Dataset.from_generator(
    simple_generator,
    output_types=(tf.float32, tf.float32),
    output_shapes=((3,), (),)
)

dataset = dataset.batch(4).prefetch(1)

model.fit(dataset)