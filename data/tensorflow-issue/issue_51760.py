from tensorflow import keras

#!/usr/bin/env python
# coding: utf-8

# Can't use tf.distribute.MirroredStrategy in srun (slurm) enviroment

# Tried with tf 2.5 and tf nightly.

import tensorflow as tf

# Force dynamic memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

tf.__version__


# op 1 . NCCL error in slurmn enviroment. Works fine inside enroot container (not submitted via srun)
strategy = tf.distribute.MirroredStrategy()

# op 2. Not using NCCL. Works.
#strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())

# op 2. Works in slurmn enviroment. Needs to be optimized
#slurm_resolver = tf.distribute.cluster_resolver.SlurmClusterResolver()
#strategy = tf.distribute.MultiWorkerMirroredStrategy(cluster_resolver=slurm_resolver)

# op 3 # Works in slurmn enviroment
#strategy = tf.distribute.MultiWorkerMirroredStrategy()


from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt


(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0


with strategy.scope():

    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    
    model.add(layers.Dense(10))
    # ADD sync bn..
    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])



history = model.fit(train_images, train_labels, epochs=10, steps_per_epoch=100)