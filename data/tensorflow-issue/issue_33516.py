import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense, Reshape
from tensorflow.keras.losses import MeanSquaredError


(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')

# Setting this here to True will break the code
TOGGLE_ERROR = True
if TOGGLE_ERROR:
    BATCH_SIZE = 1
else:
    BATCH_SIZE = 3


def map_function(train_image):
    return (train_image - 127.5) / 127.5


train_dataset = tf.data.Dataset.from_tensor_slices(train_images)
train_dataset = train_dataset.repeat()
train_dataset = train_dataset.map(map_function, tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.batch(BATCH_SIZE)
train_dataset = train_dataset.prefetch(64)


class AutoEncoder(tf.keras.Model):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.flatten = Flatten()
        self.dense_1 = Dense(128, activation="relu")
        self.dense_2 = Dense(784, activation="relu")
        self.reshape = Reshape((28, 28, 1))

    @tf.function
    def call(self, inputs):
        flatten = self.flatten(inputs)
        encoded = self.dense_1(flatten)
        decoded = self.dense_2(encoded)
        return self.reshape(decoded)


auto_encoder = AutoEncoder()
mse = MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(1e-5)


@tf.function
def train_step(batch):
    with tf.GradientTape() as tape:
        auto_encoded = auto_encoder(batch)
        loss = mse(batch, auto_encoded)

    grads = tape.gradient(loss, auto_encoder.trainable_variables)
    optimizer.apply_gradients(zip(grads, auto_encoder.trainable_variables))
    return loss


for step, image_batch in enumerate(train_dataset):
    loss = train_step(image_batch)
    if step % 1000 == 0:
        print(loss)

import tensorflow as tf
import numpy as np
import time

data = np.random.normal(size=(200,32,32))
tensor_slice_dataset = tf.data.Dataset.from_tensor_slices(data)

######################## WORKING ########################

# batch size equal to 1 works with any number
# greater or equal to 1 for number of parallel calls
map_dataset = tensor_slice_dataset.map(
    lambda x: x*x, num_parallel_calls=1)
batch_dataset = map_dataset.batch(1)

for v in batch_dataset:
    time.sleep(0.1)

##################### ALSO WORKING #####################

# batch size greater than 1 works with any number
# greater or equal to 1 and also with tf.data.experimental.AUTOTUNE
map_dataset = tensor_slice_dataset.map(
    lambda x: x*x, num_parallel_calls=tf.data.experimental.AUTOTUNE)
batch_dataset = map_dataset.batch(2)

for v in batch_dataset:
    time.sleep(0.1)

######################## BROKEN  ########################

# batch size equal to 1 DOES NOT work with
# tf.data.experimental.AUTOTUNE
map_dataset = tensor_slice_dataset.map(
    lambda x: x*x, num_parallel_calls=tf.data.experimental.AUTOTUNE)
batch_dataset = map_dataset.batch(1)

for v in batch_dataset:
    time.sleep(0.1)

batch_size = 1 * num_gpus

files = tf.data.Dataset.list_files(tfrecord_files_pattern)
if training:
    files.shuffle(buffer_size=1024)
dataset = files.interleave(tf.data.TFRecordDataset, num_parallel_calls=tf.data.experimental.AUTOTUNE)
dataset = dataset.repeat(1)
num_parallel_calls = tf.data.experimental.AUTOTUNE if batch_size > 1 else 8
dataset = dataset.map(parse_fn, num_parallel_calls=num_parallel_calls)
dataset = dataset.batch(batch_size=batch_size)
dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

...
strategy = ... # tf.distribute.Strategy here
dataset = strategy.experimental_distribute_dataset(dataset)