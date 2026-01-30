from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

"""
This script is extended from the following stock example:
https://www.tensorflow.org/beta/tutorials/distribute/keras
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPool2D
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import initializers
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Activation

import tensorflow as tf
import tensorflow_datasets as tfds
import math
import os

# Initialize GPUs
# This function must be called first
# Setting up GPU memory ussage
# https://www.tensorflow.org/beta/guide/distribute_strategy
# https://www.tensorflow.org/beta/guide/using_gpu
def init_gpus(soft_device_placement=True, log_device_placement=False, create_virtual_devices=False, memory_limit=4096):

    tf.config.set_soft_device_placement(soft_device_placement)    
    tf.debugging.set_log_device_placement(log_device_placement)

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # If there is only one GPU, create two logical virtual devices for developing
        # on a machine with only one GPU installed
        try:
            # Create 2 virtual GPUs on each physical GPU with the given memory_limit GPU memory
            if create_virtual_devices and len(gpus) == 1:
                for gpu in gpus:
                    tf.config.experimental.set_virtual_device_configuration(
                        gpu,
                        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096),
                         tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]
                    )

            else:
                # Currently, memory growth needs to be the same across GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)

        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

        # print out physical and logical GPUs
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")

    else:
        print("No visible GPU is detected...")

def scale(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255
    return image, label

# Callback for printing the LR at the end of each epoch.
class PrintLR(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print('\nLearning rate for epoch %d is %.5f' %(epoch + 1, self.model.optimizer.lr.numpy()))

# Function for decaying the learning rate.
# You can define any decay function you need.
def decay(epoch):
    if epoch < 3:
        return 1e-3
    elif epoch >= 3 and epoch < 7:
        return 1e-4
    else:
        return 1e-5

def run():
    ### get data
    init_gpus(
        log_device_placement=False,
        create_virtual_devices=True
    )

    # 
    # strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.ReductionToOneDevice())
    # strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
    # strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.NcclAllReduce())
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    # Setting with_info to True includes the metadata for the entire dataset, 
    # which is being saved here to info. Among other things, this metadata object
    # includes the number of train and test examples.
    datasets, info = tfds.load(
        name='mnist',
        with_info=True,
        as_supervised=True
    )
    mnist_train, mnist_test = datasets['train'], datasets['test']

    # You can also do info.splits.total_num_examples to get the total
    # number of examples in the dataset.
    num_train_examples = info.splits['train'].num_examples
    num_test_examples = info.splits['test'].num_examples

    #
    print("num_train_examples = %d, num_test_examples = %d" %(num_train_examples, num_test_examples))

    BUFFER_SIZE = num_train_examples
    BATCH_SIZE_PER_REPLICA = 64
    BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
    EPOCHS = 20

    # train_dataset = mnist_train.map(scale).shuffle(num_train_examples).repeat().batch(BATCH_SIZE)
    train_dataset = mnist_train.shuffle(num_train_examples).repeat().map(scale).batch(BATCH_SIZE)
    eval_dataset = mnist_test.map(scale).batch(BATCH_SIZE)

    with strategy.scope():
        ### compile
        ## Seperate the activation layer so that BatchNormalization can be added later on.
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, 3, input_shape=(28, 28, 1)),
            tf.keras.layers.Activation(activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64),
            tf.keras.layers.Activation(activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])

        model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=tf.keras.optimizers.Adam(),
            metrics=['accuracy']
        )
        model.summary()

    # print out physical and logical GPUs
    gpus = tf.config.experimental.list_physical_devices('GPU')
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")

    # Define the checkpoint directory to store the checkpoints
    checkpoint_dir = './training_checkpoints'
    # Name of the checkpoint files
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir='./logs'),
        tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix, save_weights_only=True),
        tf.keras.callbacks.LearningRateScheduler(decay),
        PrintLR()
    ]

    steps_per_epoch = math.ceil(num_train_examples / BATCH_SIZE)
    model.fit(
        train_dataset,
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    # validation
    eval_loss, eval_acc = model.evaluate(eval_dataset)

    K.clear_session()
    return

if __name__ == '__main__':
    run()