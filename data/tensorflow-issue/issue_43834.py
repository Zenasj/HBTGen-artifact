from tensorflow import keras
from tensorflow.keras import layers

3
def make_model():
  import tensorflow.keras as keras
  
  vocabulary = range(1, 11)
  return keras.Sequential([
    keras.layers.experimental.preprocessing.IntegerLookup(vocabulary=vocabulary),
    keras.layers.Embedding(len(vocabulary) + 2, 8, input_length=1),
    tf.keras.layers.Dense(1, activation='sigmoid')
  ])

for x in fitted_model.variables:
  print(type(x))

import tensorflow as tf
import horovod.tensorflow.keras as hvd

from sparkdl import HorovodRunner

def make_dataset():
  import tensorflow as tf
  from random import randrange

  dataset = tf.data.Dataset.from_tensor_slices(
    ([randrange(1, 11) for p in range(0, 10000)], [randrange(0,2) for p in range(0, 10000)])
  )
  dataset = dataset.repeat().batch(128)
  return dataset

def train():
  import tensorflow as tf
  import tensorflow.keras as keras
  import horovod.tensorflow.keras as hvd
  
  # Initialize Horovod
  hvd.init()

  # Pin GPU to be used to process local rank (one GPU per process)
  gpus = tf.config.experimental.list_physical_devices('GPU')
  for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  if gpus:
      tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

  # Build model and dataset
  dataset = make_dataset()
  model = make_model()
  # Horovod: adjust learning rate based on number of GPUs.
  scaled_lr = 0.001 * hvd.size()
  opt = tf.optimizers.Adam(scaled_lr)

  # Horovod: add Horovod DistributedOptimizer.
  opt = hvd.DistributedOptimizer(opt)

  # Horovod: Specify `experimental_run_tf_function=False` to ensure TensorFlow
  # uses hvd.DistributedOptimizer() to compute gradients.
  model.compile(
    loss=tf.losses.BinaryCrossentropy(from_logits=True),
    optimizer=opt,
    metrics=['AUC'],
    experimental_run_tf_function=False
  )

  callbacks = [
      # Horovod: broadcast initial variable states from rank 0 to all other processes.
      # This is necessary to ensure consistent initialization of all workers when
      # training is started with random weights or restored from a checkpoint.
      hvd.callbacks.BroadcastGlobalVariablesCallback(0),
  ]

  model.fit(
    dataset,
    steps_per_epoch=500 // hvd.size(),
    callbacks=callbacks,
    epochs=2,
    verbose=1 if hvd.rank() == 0 else 0
  )
  
  return model

hr = HorovodRunner(np=-1)
fitted_model = hr.run(train)

def train_local():
  # Build model and dataset
  dataset = make_dataset()
  model = make_model()
  
  opt = tf.optimizers.Adam(0.001)
  model.compile(
    loss=tf.losses.BinaryCrossentropy(from_logits=True),
    optimizer=opt,
    metrics=['AUC'],
    experimental_run_tf_function=False
  )

  model.fit(
    dataset,
    steps_per_epoch=100,
    epochs=2,
    verbose=1
  )
  
  return model