from tensorflow.keras import layers
from tensorflow.keras import optimizers

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import Xception
import numpy as np

num_samples = 1000
height = 224
width = 224
num_classes = 1000

strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    parallel_model = Xception(weights=None,
                     input_shape=(height, width, 3),
                     classes=num_classes)
    parallel_model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

# Generate dummy data.
x = np.random.random((num_samples, height, width, 3))
y = np.random.random((num_samples, num_classes))

parallel_model.summary()
parallel_model.fit(x, y, epochs=20, batch_size=16) #batch_sized changed to 16

os.environ['TF_CUDNN_DETERMINISTIC']='1'

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import Xception
import numpy as np
import os
import random

os.environ['TF_CUDNN_DETERMINISTIC']='1'

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

num_samples = 1000
height = 224
width = 224
num_classes = 1000

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(physical_devices[0:], 'GPU')

parallel_model = Xception(weights=None,
                    input_shape=(height, width, 3),
                    classes=num_classes)
parallel_model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

# Generate dummy data.
x = np.random.random((num_samples, height, width, 3))
y = np.random.random((num_samples, num_classes))

parallel_model.summary()
# This `fit` call will be distributed on 8 GPUs.
# Since the batch size is 256, each GPU will process 32 samples.
parallel_model.fit(x, y, epochs=20, batch_size=16) #batch_sized changed to 16

CUDA_ERROR_ILLEGAL_ADDRESS

MirroredStrategy

MirroredStrategy

tf.distribute.MirroredStrategy()

tf.distribute.get_strategy()

MirroredStrategy

# Import TensorFlow and TensorFlow Datasets

import tensorflow_datasets as tfds
import tensorflow as tf

import os

datasets, info = tfds.load(name='mnist', with_info=True, as_supervised=True)

mnist_train, mnist_test = datasets['train'], datasets['test']

strategy = tf.distribute.MirroredStrategy()

num_train_examples = info.splits['train'].num_examples
num_test_examples = info.splits['test'].num_examples

BUFFER_SIZE = 10000

BATCH_SIZE_PER_REPLICA = 64
BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

def scale(image, label):
  image = tf.cast(image, tf.float32)
  image /= 255

  return image, label

train_dataset = mnist_train.map(scale).cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
eval_dataset = mnist_test.map(scale).batch(BATCH_SIZE)

with strategy.scope():
  model = tf.keras.Sequential([
      tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(64, activation='relu'),
      tf.keras.layers.Dense(10)
  ])

  model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                optimizer=tf.keras.optimizers.Adam(),
                metrics=['accuracy'])

# Define the checkpoint directory to store the checkpoints

checkpoint_dir = './training_checkpoints'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

# Function for decaying the learning rate.
# You can define any decay function you need.
def decay(epoch):
  if epoch < 3:
    return 1e-3
  elif epoch >= 3 and epoch < 7:
    return 1e-4
  else:
    return 1e-5

# Callback for printing the LR at the end of each epoch.
class PrintLR(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    print('\nLearning rate for epoch {} is {}'.format(epoch + 1,
                                                      model.optimizer.lr.numpy()))

callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir='./logs'),
    tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix,
                                       save_weights_only=True),
    tf.keras.callbacks.LearningRateScheduler(decay),
    PrintLR()
]

model.fit(train_dataset, epochs=12, callbacks=callbacks)

strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())

Nccl