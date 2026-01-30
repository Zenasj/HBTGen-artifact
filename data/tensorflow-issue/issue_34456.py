import math
import random
from tensorflow.keras import layers
from tensorflow.keras import models

# imports
import time

import numpy as np
import tensorflow as tf
from tensorflow.python.keras import callbacks as cbks
from tensorflow.keras.callbacks import Callback
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.engine import training_utils
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.utils import data_utils
from tensorflow.python.keras.utils import generic_utils
from tqdm import tqdm_notebook

# helper function (taken from https://github.com/tensorflow/tensorflow/blob/r2.0/tensorflow/python/keras/engine/training_generator.py#L500)
def _make_enqueued_generator(generator,
                             workers=1,
                              use_multiprocessing=False,
                             max_queue_size=10,
                             shuffle=False):    
    enqueuer = data_utils.OrderedEnqueuer(
        generator, use_multiprocessing=use_multiprocessing, shuffle=shuffle)
    enqueuer.start(workers=workers, max_queue_size=max_queue_size)
    output_generator = enqueuer.get()
    return output_generator, enqueuer

# My silly callback
class Noise(Callback):
     def on_batch_end(self, batch, logs={}):
        image_shape = [1, 2**7, 2**7, 1]
        noise = np.random.normal(scale=1.0, size=image_shape)

# My data
batch_size = 8
n_samples_train = 720
x = np.random.rand(n_samples_train, 256, 256, 1)
im_gen_train = ImageDataGenerator().flow(x, batch_size=batch_size)


# My training set up (to mimic https://github.com/tensorflow/tensorflow/blob/r2.0/tensorflow/python/keras/engine/training_generator.py#L41)
data = im_gen_train
steps_per_epoch = int(n_samples_train / batch_size)
epochs = 20
max_queue_size=35
workers=35
use_multiprocessing=True
shuffle=False
initial_epoch=0
mode=1
steps_name='steps'
noise_cb = Noise()
noise_cb.on_train_batch_end = noise_cb.on_batch_end
callbacks=[noise_cb]

generator, enqueuer = _make_enqueued_generator(
    im_gen_train,
    workers=workers,
    use_multiprocessing=use_multiprocessing,
    max_queue_size=max_queue_size,
    shuffle=shuffle)

callbacks = cbks.configure_callbacks(
    callbacks,
    Model(),
    do_validation=False,
    epochs=epochs,
    steps_per_epoch=steps_per_epoch,
    batch_size=batch_size,
    samples=n_samples_train,
    verbose=0,  # Handle ProgBar as part of Callbacks once hooks are ready.
    mode=mode,
)
callbacks._call_begin_hook(mode)

for epoch in tqdm_notebook(range(initial_epoch, epochs)):
    callbacks.on_epoch_begin(epoch, {})

    for step in tqdm_notebook(range(steps_per_epoch), leave=False):
        callbacks._call_batch_hook('train', 'begin', step, {})
        batch_data = next(generator)
        
        # I don't actually train a model, so I just sleep for this time, this would be the backprop
        time.sleep(0.1)
        callbacks._call_batch_hook('train', 'end', step, {})

# imports
import time

from keras_tqdm import TQDMNotebookCallback
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import Input, Conv2D, Lambda, concatenate
from tensorflow.python.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import  Sequence


# My silly callback
class Noise(Callback):
     def on_batch_end(self, batch, logs={}):
        image_shape = [1, 2**7, 2**7, 1]
        noise = np.random.normal(scale=1.0, size=image_shape)
        
# my metrics
def keras_psnr(y_true, y_pred):
    max_pixel = tf.math.reduce_max(y_true)
    min_pixel = tf.math.reduce_min(y_true)
    return tf.image.psnr(y_true, y_pred, max_pixel - min_pixel)

def keras_ssim(y_true, y_pred):
    max_pixel = tf.math.reduce_max(y_true)
    min_pixel = tf.math.reduce_min(y_true)
    return tf.image.ssim(y_true, y_pred, max_pixel - min_pixel)

# My data
class MergedGenerators(Sequence):
    def __init__(self, *generators):
        self.generators = generators
        # TODO add a check to verify that all generators have the same length

    def __len__(self):
        return len(self.generators[0])

    def __getitem__(self, index):
        return tuple([generator[index] for generator in self.generators])

batch_size = 8
n_samples_train = 720
size = 256
x = np.random.rand(n_samples_train, size, size, 1)
im_gen_train_1 = ImageDataGenerator().flow(x, batch_size=batch_size, seed=0)
im_gen_train_2 = ImageDataGenerator().flow(x, batch_size=batch_size, seed=0)
im_gen_train = MergedGenerators(im_gen_train_1, im_gen_train_2)

# my fake model
im = Input((None, None, 1))
conv = Conv2D(256, 3, padding='same')(im)
conv = Conv2D(256, 3, padding='same')(conv)
conv = Conv2D(1, 3, padding='same')(conv)
ident = Lambda(lambda x: x)(conv)
model = Model(im, ident)
model.compile(loss='mse', optimizer='adam', metrics=[keras_psnr, keras_ssim])
print(model.summary(line_length=150))

# My training set up
noise_cb = Noise()
noise_cb.on_train_batch_end = noise_cb.on_batch_end
tqdm_cb = TQDMNotebookCallback(metric_format="{name}: {value:e}")
tqdm_cb.on_train_batch_begin = tqdm_cb.on_batch_begin
tqdm_cb.on_train_batch_end = tqdm_cb.on_batch_end
model.fit_generator(
    im_gen_train,
    steps_per_epoch=int(n_samples_train / batch_size), 
    epochs=20,
    max_queue_size=35,
    workers=35,
    use_multiprocessing=True,
    shuffle=False,
    callbacks=[noise_cb, tqdm_cb],
    verbose=0,
)