from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf
import tensorflow_datasets as tfds
import time
import datetime

import argparse
import sys
import shutil
import time
import os
import numpy as np

from functools import partial
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, GlobalAveragePooling2D
from tensorflow.keras import Model
from tensorflow.keras.datasets.fashion_mnist import load_data
import tensorflow_datasets as tfds
import read_params
from train_config import configure_model, configure_optimizer, configure_lossfunc
from datasets.readtf_utils.dataset import get_dataset 
from datasets.readtf_utils.dataset import _parse_fn

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)


num_epochs = 5
batch_size_per_replica = 128
learning_rate = 0.001
setting_GPUs_num = 8
devices = ["/gpu:"+str(i) for i in range(setting_GPUs_num)]

strategy = tf.distribute.MirroredStrategy(devices)
GPUs_num = strategy.num_replicas_in_sync
print('Number of devices: %d' % GPUs_num)  # 输出设备数量

batch_size = batch_size_per_replica

# 载入数据集并预处理
def resize(image, label):
    image = tf.image.resize(image, [224, 224]) / 255.0
    return image, label



class MyCustomCallback(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.train_begin = time.time()
        self.times = []
        # print('trian begins at {}'.format(self.train_begin))
        # d

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_begin = time.time()
        # print('epoch: {} begins at {}'.format(epoch, self.epoch_begin))

    def on_epoch_end(self, epoch, logs=None):
        self.epoch_end = time.time()
        self.epoch = epoch
        # print('epoch: {} ends at {}'.format(epoch, self.epoch_end))
        self.times.append(self.epoch_end-self.epoch_begin)
        print(" epoch: {} takes: {}".format(epoch, self.epoch_end-self.epoch_begin))

    def on_train_end(self, epoch, logs=None):
        self.train_end = time.time()
        # print('training takes {} secs/epoch: '.format((self.train_end - self.train_begin)/self.epoch))
        print('training takes average {:.2f} secs/epoch'.format(sum(self.times[1::]) / (self.epoch)))



tfrecords_dir = "/data121/lijiayuan/test/classify_flowers/datasets/"
dataset, _ = get_dataset(tfrecords_dir, subset="train", batch_size=batch_size)


if GPUs_num == 1:
    model = tf.keras.applications.MobileNetV2()
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        metrics=[tf.keras.metrics.sparse_categorical_accuracy]
    )
else:
    with strategy.scope():
        model = tf.keras.applications.MobileNetV2()
        model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
            loss=tf.keras.losses.sparse_categorical_crossentropy,
            metrics=[tf.keras.metrics.sparse_categorical_accuracy]
            # run_eagerly=False
        )


start = time.time()
model.fit(dataset, epochs=num_epochs, callbacks=[MyCustomCallback()])
end = time.time()

print("{} GPUs takes {:.2f} secs/epoch = {:.2f} mins/epoch".format(strategy.num_replicas_in_sync, 
                                                                    (end-start)/num_epochs, 
                                                                    (end-start)/60/num_epochs))

from __future__ import absolute_import, division, print_function, unicode_literals

# Import TensorFlow
import tensorflow as tf

# Helper libraries
import numpy as np
import os
import time
import tensorflow_datasets as tfds
from datasets.readtf_utils.dataset import get_dataset 
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, GlobalAveragePooling2D



os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)



def resize(image, label):
    image = tf.image.resize(image, [224, 224]) / 255.0
    return image, label

def assemble_model(num_classes, model_name='MobileNetV2'):
    import tensorflow as tf 
    base_model = tf.keras.applications.ResNet50(input_shape=(224,224,3),
                                                    weights='imagenet',
                                                    include_top=False)
    model = tf.keras.Sequential([
                                base_model,
                                GlobalAveragePooling2D(),
                                Dense(num_classes, activation='softmax')
                                ])
    model.trainable = True
    return model


print(tf.__version__)

setting_GPUs_num = 8
devices = ["/gpu:"+str(i) for i in range(setting_GPUs_num)]

strategy = tf.distribute.MirroredStrategy(devices)
print ('Number of devices: {}'.format(strategy.num_replicas_in_sync))
BATCH_SIZE_PER_REPLICA = 256
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

EPOCHS = 5


##-----dataset------##
tfrecords_dir = "/data121/lijiayuan/test/classify_flowers/datasets"
train_ds, classes_num = get_dataset(tfrecords_dir, subset="train", batch_size=GLOBAL_BATCH_SIZE)


train_ds = strategy.experimental_distribute_dataset(train_ds)






with strategy.scope():
  # Set reduction to `none` so we can do the reduction afterwards and divide by
  # global batch size.
  loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
      reduction=tf.keras.losses.Reduction.NONE)
  # or loss_fn = tf.keras.losses.sparse_categorical_crossentropy
  def compute_loss(labels, predictions):
    per_example_loss = loss_object(labels, predictions)
    return tf.nn.compute_average_loss(per_example_loss, global_batch_size=GLOBAL_BATCH_SIZE)


with strategy.scope():

  train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
      name='train_accuracy')

# model and optimizer must be created under `strategy.scope`.
with strategy.scope():
  model = assemble_model(num_classes=classes_num)
  optimizer = tf.keras.optimizers.Adam()
  checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)

with strategy.scope():
  def train_step(inputs):
    images, labels = inputs

    with tf.GradientTape() as tape:
      predictions = model(images, training=True)
      loss = compute_loss(labels, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_accuracy.update_state(labels, predictions)
    return loss 




with strategy.scope():

  @tf.function
  def distributed_train_step(dataset_inputs):
    per_replica_losses = strategy.experimental_run_v2(train_step,
                                                      args=(dataset_inputs,))
    return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                           axis=None)
 
 
  times = []
  for epoch in range(EPOCHS):
    # TRAIN LOOP

    total_loss = 0.0
    num_batches = 0
    epoch_start = time.time()
    for x in train_ds:
      total_loss += distributed_train_step(x)
      num_batches += 1
    train_loss = total_loss / num_batches
    epoch_end = time.time()
    
    if epoch != 0:
      times.append(epoch_end-epoch_start)
    



    template = ("Epoch {}, Loss: {:.2f}, Accuracy: {:.2f}, "
                " Takes: {:.2f}")
    print (template.format(epoch+1, train_loss,
                           train_accuracy.result()*100, 
                           epoch_end-epoch_start))

    train_accuracy.reset_states()
  print("{} GPUs takes average {:.2f} secs".format(setting_GPUs_num, 
                                                    sum(times)/(EPOCHS-1)))