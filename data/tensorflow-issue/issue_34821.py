from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import tensorflow_datasets as tfds
import tensorflow as tf
from absl import app, flags
import os
import json

tfds.disable_progress_bar()
FLAGS = flags.FLAGS

flags.DEFINE_string('input_data_path', default='hdfs://30.78.5.52:9000/data/public/dataset/tensorflow_datasets', help='HDFS Input data path')
flags.DEFINE_string('checkpoint_dir', default=None, help='HDFS checkpoint dir')

BUFFER_SIZE = 10000
BATCH_SIZE = 64

def make_datasets_unbatched():
  # Scaling MNIST data from (0, 255] to (0., 1.]
  def scale(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255
    return image, label

  datasets, info = tfds.load(name='mnist',
                            with_info=True,
                            as_supervised=True,
                            download=False,
                            data_dir=FLAGS.input_data_path)

  return datasets['train'].map(scale).cache().shuffle(BUFFER_SIZE)


def build_and_compile_cnn_model():
  model = tf.keras.Sequential([
      tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(64, activation='relu'),
      tf.keras.layers.Dense(10, activation='softmax')
  ])
  model.compile(
      loss=tf.keras.losses.sparse_categorical_crossentropy,
      optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
      metrics=['accuracy'])
  return model

def main(argv):
  strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

  NUM_WORKERS = 2
  # Here the batch size scales up by number of workers since
  # `tf.data.Dataset.batch` expects the global batch size. Previously we used 64,
  # and now this becomes 128.
  GLOBAL_BATCH_SIZE = 64 * NUM_WORKERS
  # Replace the `filepath` argument with a path in the file system
  # accessible by all workers.
  callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath=FLAGS.checkpoint_dir)]
  with strategy.scope():
    # Creation of dataset, and model building/compiling need to be within
    # `strategy.scope()`.
    train_datasets = make_datasets_unbatched().batch(GLOBAL_BATCH_SIZE)
    multi_worker_model = build_and_compile_cnn_model()
  multi_worker_model.fit(x=train_datasets, epochs=2, callbacks=callbacks)



if __name__ == '__main__':
  app.run(main)