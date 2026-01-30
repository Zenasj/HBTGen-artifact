from tensorflow import keras
from tensorflow.keras import layers

import os
import fnmatch
import random
import time
import tensorflow as tf
import numpy as np
#import librosa

NUM_EPOCHS = 2
BATCH_SIZE = 16
FRAME_SIZE = 64
NUM_BATCHES = 4
SEQ_LEN = 1024
OVERLAP = FRAME_SIZE
NUM_SAMPS = 128000 + OVERLAP

##################### DATASET #####################

def find_files(directory, pattern='*.wav'):
    '''Recursively finds all files matching the pattern.'''
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            files.append(os.path.join(root, filename))
    random.shuffle(files)
    return files

# This can load audio files (preferably of only a few seconds long!), but I have commented out that code
# and replaced with a line that simply generates some arrays of random 'samples'...
def load_audio(files, batch_size):
    #for filename in files:
    for filename in range(0, 350):
        #(audio, _) = librosa.load(filename, sr=None, mono=True)
        audio = np.random.randint(0, 256, size=(batch_size * NUM_SAMPS))
        audio = audio.reshape(-1, 1)
        print("Loading corpus entry {}".format(filename))
        yield audio

def pad_batch(batch, batch_size, seq_len, overlap):
    num_samps = ( int(np.floor(len(batch[0]) / float(seq_len))) * seq_len )
    zeros = np.zeros([batch_size, overlap, 1], dtype='float32')
    return tf.concat([zeros, batch[:, :num_samps, :]], axis=1)

def get_subseqs(dataset, batch_size, seq_len, overlap):
    for batch in dataset:
        num_samps = len(batch[0])
        for i in range(overlap, num_samps, seq_len):
            x = batch[:, i-overlap : i+seq_len]
            y = x[:, overlap : overlap+seq_len]
            yield (x, y)

def get_dataset(files=None, num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, overlap=OVERLAP,
                seq_len=SEQ_LEN, drop_remainder=False):
    dataset = tf.data.Dataset.from_generator(
        lambda: load_audio(files, batch_size),
        output_types=tf.float32,
        output_shapes=((None, 1))
    )
    dataset = dataset.repeat(num_epochs).batch(batch_size, drop_remainder)
    dataset = dataset.map(lambda batch: tf.py_function(
        func=pad_batch, inp=[batch, batch_size, seq_len, overlap], Tout=tf.float32
    ))
    return tf.data.Dataset.from_generator(
        lambda: get_subseqs(dataset, batch_size, seq_len, overlap),
        output_types=(tf.int32, tf.int32),
        output_shapes=(
            (batch_size, seq_len + overlap, 1),
            (batch_size, seq_len, 1)))

# If we use the following there is NO hang...

def get_rand_seqs(num_samps, num_batches=NUM_BATCHES, batch_size=BATCH_SIZE, overlap=OVERLAP, seq_len=SEQ_LEN):
    time.sleep(0.03)
    for _ in range(0, num_batches):
        batch = np.random.randint(0, 256, size=(batch_size * num_samps))
        batch = batch.reshape((batch_size, num_samps, 1))
        for i in range(overlap, num_samps, seq_len):
            x = batch[:, i-overlap : i+seq_len]
            y = x[:, overlap : overlap+seq_len]
            yield (x, y)

def get_dataset_NO_HANG(num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, overlap=OVERLAP, seq_len=SEQ_LEN):
    dataset = tf.data.Dataset.from_generator(
        lambda: get_subseqs(NUM_SAMPS),
        output_types=(tf.int32, tf.int32),
        output_shapes=(
            (batch_size, seq_len + overlap, 1),
            (batch_size, seq_len, 1)))
    dataset = dataset.repeat(num_epochs)
    return dataset

##################### MODEL #####################

# This is just a dummy model, not my real one, although I do make extensive use of RNNs...
class TestModel(tf.keras.Model):

    def __init__(self, frame_size=FRAME_SIZE, dim=1024):
        super(TestModel, self).__init__()
        self.frame_size = frame_size
        self.q_levels = 256
        self.dim = dim
        self.num_lower_tier_frames = 4
        self.input_expand = tf.keras.layers.Dense(self.dim)
        self.rnn = tf.keras.layers.GRU(self.dim, return_sequences=True, stateful=True)
        self.upsample = tf.Variable(
            tf.initializers.GlorotNormal()(
                shape=[self.num_lower_tier_frames, self.dim, self.dim]),
            name="upsample",
        )
        self.out = tf.keras.layers.Dense(self.q_levels, activation='relu')

    def train_step(self, data):
        (x, y) = data
        with tf.GradientTape() as tape:
            raw_output = self(x, training=True)
            prediction = tf.reshape(raw_output, [-1, self.q_levels])
            target = tf.reshape(y, [-1])
            loss = self.compiled_loss(
                target,
                prediction,
                regularization_losses=self.losses)
        grads = tape.gradient(loss, self.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 5.0)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.compiled_metrics.update_state(target, prediction)
        return {metric.name: metric.result() for metric in self.metrics}

    def call(self, input):
        batch_size = tf.shape(input)[0]
        input = tf.cast(input[:, : -self.frame_size, :], tf.float32)
        frames = tf.reshape(input, [
            batch_size,
            tf.shape(input)[1] // self.frame_size,
            self.frame_size
        ])
        num_steps = tf.shape(frames)[1]
        frames = self.input_expand(frames)
        hidden = self.rnn(frames)
        output_shape = [
            batch_size,
            num_steps * self.num_lower_tier_frames,
            self.dim
        ]
        outputs = tf.nn.conv1d_transpose(
            hidden,
            self.upsample,
            strides=self.num_lower_tier_frames,
            output_shape=output_shape,
        )
        return self.out(tf.transpose(outputs, perm=(0,2,1)))

##################### TRAINING #####################

#files = find_files('path/to/wavs') # Leave this as is, no audio needs to be loaded
train_dataset = get_dataset() # This hangs on the HTC, but get_dataset_NO_HANG does not...

model = TestModel()

opt = tf.optimizers.Adam(learning_rate=0.001, epsilon=1e-4)
compute_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')
model.compile(optimizer=opt, loss=compute_loss, metrics=[train_accuracy])

model.fit(
    train_dataset,
    epochs=NUM_EPOCHS,
    steps_per_epoch=500,
    shuffle=False
)

for (i, (x, y)) in enumerate(train_dataset):
    print(i)
    time.sleep(0.2)
    print(tf.shape(x))
    print(tf.shape(y))

def get_dataset(files=None, num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, overlap=OVERLAP,
                seq_len=SEQ_LEN, drop_remainder=False):
    dataset = tf.data.Dataset.from_generator(
        lambda: load_audio(files, batch_size),
        output_types=tf.float32,
        output_shapes=((None, 1))
    )
    dataset = dataset.repeat(num_epochs).batch(batch_size, drop_remainder)
    return dataset.map(lambda batch: tf.py_function(
        func=pad_batch, inp=[batch, batch_size, seq_len, overlap], Tout=tf.float32
    ))

for (i, x) in enumerate(train_dataset):
    print(i)
    time.sleep(0.2)
    print(tf.shape(x))

def get_dataset(files=None, num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, overlap=OVERLAP,
                  seq_len=SEQ_LEN, drop_remainder=False):
    dataset = tf.data.Dataset.from_generator(
        lambda: load_audio(files, batch_size),
        output_types=tf.float32,
        output_shapes=((None, 1))
    )
    dataset = dataset.repeat(num_epochs).batch(batch_size, drop_remainder)
    return dataset.map(lambda batch: tf.py_function(
        func=pad_batch, inp=[batch, batch_size, seq_len, overlap], Tout=tf.float32
    ))

def ds_iter(dataset, seq_len=SEQ_LEN, overlap=OVERLAP):
    for batch in dataset:
        print('ds_iter')
        num_samps = len(batch[0])
        for i in range(overlap, num_samps, seq_len):
            x = batch[:, i-overlap : i+seq_len]
            y = x[:, overlap : overlap+seq_len]
            yield (x, y)

dataset = get_dataset(drop_remainder=True)
gen = ds_iter(dataset)
for (i, (x, y)) in enumerate(gen):
    print(i)
    time.sleep(0.2)
    print(tf.shape(x))
    print(tf.shape(y))