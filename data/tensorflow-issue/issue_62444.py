from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

validation_steps

60,000

10,000

num_epochs=2

batch_size=8

(train, test), info = tfds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

x_generator = train.batch(batch_size).as_numpy_iterator()
v_generator = test.batch(batch_size).as_numpy_iterator()   # using 'test' for validation here

num_epochs = 2
batch_size = 8

num_x_batches_per_epoch = int(np.floor(num_total_train_records / batch_size / num_epochs))
num_v_batches_per_epoch = int(np.floor(num_total_test_records / batch_size / num_epochs)) 
print(num_x_batches_per_epoch, num_v_batches_per_epoch)
# ---
# show 3750 625

tf.keras.Model.fit(validation_steps=625)

model.fit(
    x=x_generator ,
    epochs=num_epochs,
    batch_size=batch_size,
    steps_per_epoch=num_x_batches_per_epoch,
    validation_data=v_generator,
    validation_steps=num_v_batches_per_epoch,
    validation_batch_size=batch_size
)

import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds


(train, test), info = tfds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)


def f(image, label):
    return 1

num_total_train_records = len(list(
    train.map(f)
))
num_total_test_records = len(list(
    test.map(f)
))
print(num_total_train_records, num_total_test_records)

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10)
])
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

num_epochs = 2
batch_size = 8

num_x_batches_per_epoch = int(np.floor(num_total_train_records / batch_size / num_epochs))
num_v_batches_per_epoch = int(np.floor(num_total_test_records / batch_size / num_epochs)) 
print(num_x_batches_per_epoch, num_v_batches_per_epoch)
# ---
# will show 3750 625


x_generator = train.batch(batch_size).as_numpy_iterator()
v_generator = test.batch(batch_size).as_numpy_iterator()

model.fit(
    x=x_generator ,
    epochs=num_epochs,
    batch_size=batch_size,
    steps_per_epoch=num_x_batches_per_epoch,
    validation_data=v_generator,
    validation_steps=num_v_batches_per_epoch,
    validation_batch_size=batch_size
)

num_v_batches_per_epoch = int(np.floor(num_total_test_records / batch_size / num_epochs)) -1  # Cuase ran out of data without -1

tf.data.Dataset

import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds


(train, test), info = tfds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)


def f(image, label):
    return 1

num_total_train_records = len(list(
    train.map(f)
))
num_total_test_records = len(list(
    test.map(f)
))
print(num_total_train_records, num_total_test_records)

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10)
])
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

num_epochs = 2
batch_size = 8

num_x_batches_per_epoch = int(np.floor(num_total_train_records / batch_size / num_epochs))
num_v_batches_per_epoch = int(np.floor(num_total_test_records / batch_size / num_epochs)) 
print(num_x_batches_per_epoch, num_v_batches_per_epoch)
# ---
# will show 3750 625


x_generator = train.batch(batch_size).as_numpy_iterator()
v_generator = test.batch(batch_size).as_numpy_iterator()

model.fit(
    x=x_generator ,
    epochs=num_epochs,
    batch_size=batch_size,
    steps_per_epoch=num_x_batches_per_epoch,
    validation_data=v_generator,
    validation_steps=num_v_batches_per_epoch,
    validation_batch_size=batch_size
)

steps_per_epoch = len(X_train)//batch_size

validation_steps = len(X_test)//batch_size