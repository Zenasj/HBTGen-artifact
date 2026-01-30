import random
from tensorflow.keras import layers

model.compile(optimizer="adam",
                 loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                 metrics=["accuracy"])

checkpoint_dir = 'training/'
checkpoint_path = os.path.join(checkpoint_dir,"cp-{epoch:04d}.ckpt")
checkpoint_path = os.path.join(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    verbose=1,
    save_weights_only=True,
    save_freq=1000)

model.load_weights(tf.train.latest_checkpoint(os.path.dirname(checkpoint_dir))).expect_partial()

from alfred.dl.tf.common import mute_tf
mute_tf()
import pathlib
import tensorflow as tf
from tensorflow import keras
from datetime import datetime
import numpy as np
import os
from pynvml import *
import multiprocessing
import argparse
import time 

os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'

parser = argparse.ArgumentParser(description="Image classification")
parser.add_argument('--input_dir',type=str,default="data_dir",help='input image directory,default is data_dir')
parser.add_argument('--epochs',type=int,default=500,help='epochs times,default is 500')
parser.add_argument('--batch_size',type=int,default=64,help='batch size,default is 64')
parser.add_argument('--from_cp',type=bool,default=False,help='training from last checkpoint')
args = parser.parse_args()


l_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
gpus = tf.config.experimental.list_physical_devices('GPU')
nvmlInit()
device_count = nvmlDeviceGetCount()
for i in range(device_count):
    handle = nvmlDeviceGetHandleByIndex(i)
    mem = nvmlDeviceGetMemoryInfo(handle).total // 1024 //1024
    tf.config.experimental.set_virtual_device_configuration(gpus[i], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=mem*0.9)])
    print('GPU Total Memory:{} limit: {}'.format(mem,mem*0.9))

cpu_core = multiprocessing.cpu_count()
print('CPU total core: {}'.format(cpu_core))

IMG_HEIGHT = 224
IMG_WIDTH = 224
NUM_WORKERS = cpu_core
batch_size = args.batch_size
epochs = args.epochs
data_dir = args.input_dir
from_cp = args.from_cp

AUTOTUNE = tf.data.experimental.AUTOTUNE



#gpus = tf.config.experimental.list_physical_devices('GPU')
#tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=mem_0*0.9)])
#tf.config.experimental.set_virtual_device_configuration(gpus[1], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=mem_0*0.9)])

mirrored_strategy = tf.distribute.MirroredStrategy()

train_dir = pathlib.Path(os.path.join(data_dir,'train'))
valid_dir = pathlib.Path(os.path.join(data_dir,'valid'))
image_count = len(list(train_dir.glob('*/*.jpg'))) + len(list(valid_dir.glob('*/*.jpg')))
class_names = np.array(sorted([item.name for item in train_dir.glob('*')]))
TRAIN_STEPS_PER_EPOCH = np.ceil((image_count*0.8/batch_size)-1)
VAL_STEPS_PER_EPOCH = np.ceil((image_count*0.2/batch_size)-1)

def pre_process_image(image):

    image = tf.image.random_flip_left_right(image)
    #image = tf.image.random_hue(image, max_delta=0.05)
    #image = tf.image.random_contrast(image, lower=0.3, upper=1.0)
    image = tf.image.random_brightness(image, max_delta=0.2)
    #image = tf.image.random_saturation(image, lower=0.0, upper=2.0)
    #image = tf.image.rot90(image, np.random.randint(1,4))
    return image


def get_label(file_path):
    # convert the path to a list of path components
    parts = tf.strings.split(file_path, os.path.sep)
    # The second to last is the class-directory
    one_hot = parts[-2] == class_names
    print(class_names)
    # Integer encode the label
    return tf.argmax(one_hot)

def decode_img(img):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)
    #img = pre_process_image(img)
    # resize the image to the desired size
    return tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])

def process_path(file_path):
    label = get_label(file_path)
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, label

def get_dataset(subset):
    if subset == 'train':
        list_ds = tf.data.Dataset.list_files(str(train_dir/'*/*'),shuffle=False)
    if subset == 'valid':
        list_ds = tf.data.Dataset.list_files(str(valid_dir/'*/*'),shuffle=False)
    list_ds = list_ds.shuffle(image_count, reshuffle_each_iteration=False)

    print(subset)
    '''
    log_txt = open(subset+'.txt', 'w')
    for d in ds:
        log_txt.write(str(d) + '\n')
    log_txt.close()
    '''
    ds = list_ds.map(process_path,num_parallel_calls=cpu_core)
    #ds = ds.cache()
    ds = ds.shuffle(buffer_size=1000)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=cpu_core).repeat()
    return ds

train_ds = get_dataset('train')
valid_ds = get_dataset('valid')

#print(AUTOTUNE)
#train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
#valid_ds = valid_ds.cache().prefetch(buffer_size=AUTOTUNE)


with mirrored_strategy.scope():
    tinydarknet = keras.Sequential([
        keras.layers.Conv2D(16, (3, 3), strides=[1, 1], padding="same", input_shape=(IMG_HEIGHT,IMG_WIDTH, 3)),
        keras.layers.BatchNormalization(),
        keras.layers.LeakyReLU(alpha=0.1),
        keras.layers.MaxPooling2D((2, 2), strides=(2, 2)),
        keras.layers.Conv2D(32, (3, 3), strides=[1, 1], padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.LeakyReLU(alpha=0.1),
        keras.layers.MaxPooling2D((2, 2), strides=(2, 2)),
        keras.layers.Conv2D(16, (1, 1), strides=[1, 1], padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.LeakyReLU(alpha=0.1),
        keras.layers.Conv2D(128, (3, 3), strides=[1, 1], padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.LeakyReLU(alpha=0.1),
        keras.layers.Conv2D(16, (1, 1), strides=[1, 1], padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.LeakyReLU(alpha=0.1),
        keras.layers.Conv2D(128, (3, 3), strides=[1, 1], padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.LeakyReLU(alpha=0.1),
        keras.layers.MaxPooling2D((2, 2), strides=(2, 2)),
        keras.layers.Conv2D(32, (1, 1), strides=[1, 1], padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.LeakyReLU(alpha=0.1),
        keras.layers.Conv2D(256, (3, 3), strides=[1, 1], padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.LeakyReLU(alpha=0.1),
        keras.layers.Conv2D(32, (1, 1), strides=[1, 1], padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.LeakyReLU(alpha=0.1),
        keras.layers.Conv2D(256, (3, 3), strides=[1, 1], padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.LeakyReLU(alpha=0.1),
        keras.layers.MaxPooling2D((2, 2), strides=(2, 2)),
        keras.layers.Conv2D(64, (1, 1), strides=[1, 1], padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.LeakyReLU(alpha=0.1),
        keras.layers.Conv2D(512, (3, 3), strides=[1, 1], padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.LeakyReLU(alpha=0.1),
        keras.layers.Conv2D(64, (1, 1), strides=[1, 1], padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.LeakyReLU(alpha=0.1),
        keras.layers.Conv2D(512, (3, 3), strides=[1, 1], padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.LeakyReLU(alpha=0.1),
        keras.layers.Conv2D(128, (1, 1), strides=[1, 1], padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.LeakyReLU(alpha=0.1),
        keras.layers.Conv2D(1000, (1, 1)),
        keras.layers.BatchNormalization(),
        keras.layers.AveragePooling2D(),
        keras.layers.Flatten(),
        keras.layers.Dense(1,activation="sigmoid")
    ])


    tinydarknet.compile(optimizer="adam",
                 loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                 metrics=["accuracy"])

logs = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tboard_callback = tf.keras.callbacks.TensorBoard(log_dir = logs,
                                                 histogram_freq = 1,
                                                 profile_batch = '500,520')

#save checkpoint

checkpoint_dir = 'training/'
checkpoint_path = os.path.join(checkpoint_dir,"cp-{epoch:04d}.ckpt")
checkpoint_path = os.path.join(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    verbose=1,
    save_weights_only=True,
    save_freq=1000)

if from_cp:
    print("checkpoint_path::" + os.path.dirname(checkpoint_dir))
    tinydarknet.load_weights(tf.train.latest_checkpoint(os.path.dirname(checkpoint_dir))).expect_partial()
    #checkpoint = tf.train.Checkpoint(myModel=tinydarknet)
    #checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

history = tinydarknet.fit(
    train_ds,
    steps_per_epoch=TRAIN_STEPS_PER_EPOCH,
    epochs=epochs,
    validation_data=valid_ds,
    validation_steps=VAL_STEPS_PER_EPOCH,
    workers=NUM_WORKERS,
    callbacks = [tboard_callback,cp_callback]
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss=history.history['loss']
val_loss=history.history['val_loss']

epochs_range = range(epochs)



#save model
tinydarknet.save(l_time+"-keras_model")

converter = tf.lite.TFLiteConverter.from_keras_model(tinydarknet)
tflite_model = converter.convert()

with open(l_time+"-tinydarknet.tflite", "w+b") as fp:
    fp.write(tflite_model)
    fp.flush()