import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import os
os.environ['TF_KERAS'] = '1'

from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from heatmaps import *
import numpy as np
from keras_radam import RAdam

image_size = 200
## output shape is the same as input
n = 32 * 5
nClasses = 6
nfmp_block1 = 64
nfmp_block2 = 128
batch_size = 64

IMAGE_ORDERING = "channels_last"
img_input = tf.keras.Input(shape=(image_size, image_size, 3))

# Encoder Block 1
x = Conv2D(nfmp_block1, (3, 3), activation='relu', padding='same', name='block1_conv1', data_format=IMAGE_ORDERING)(
    img_input)
x = Conv2D(nfmp_block1, (3, 3), activation='relu', padding='same', name='block1_conv2', data_format=IMAGE_ORDERING)(x)
block1 = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool', data_format=IMAGE_ORDERING)(x)

# Encoder Block 2
x = Conv2D(nfmp_block2, (3, 3), activation='relu', padding='same', name='block2_conv1', data_format=IMAGE_ORDERING)(
    block1)
x = Conv2D(nfmp_block2, (3, 3), activation='relu', padding='same', name='block2_conv2', data_format=IMAGE_ORDERING)(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool', data_format=IMAGE_ORDERING)(x)

## bottleneck
o = (Conv2D(n, (int(image_size / 4), int(image_size / 4)),
            activation='relu', padding='same', name="bottleneck_1", data_format=IMAGE_ORDERING))(x)
o = (Conv2D(n, (1, 1), activation='relu', padding='same', name="bottleneck_2", data_format=IMAGE_ORDERING))(o)

## Decoder Block
## upsampling to bring the feature map size to be the same as the input image i.e., heatmap size
output = Conv2DTranspose(nClasses, kernel_size=(4, 4), strides=(4, 4), use_bias=False, name='upsample_2',
                         data_format=IMAGE_ORDERING)(o)

## Reshaping is necessary to use sample_weight_mode="temporal" which assumes 3 dimensional output shape
## See below for the discussion of weights
output = Reshape((image_size * image_size * nClasses, 1))(output)
model = tf.keras.Model(img_input, output)
model.summary()

radam = RAdam(total_steps=10000, warmup_proportion=0.1, min_lr=1e-5)
model.compile(optimizer=radam, loss='mse', sample_weight_mode="temporal")

data_folder = 'data'
id2filename, filename2id, annotated_images = dataloader.get_image_annotations(data_folder)
df = dataloader.get_annotation_dataframe(id2filename, annotated_images)
msk = np.random.rand(len(df)) < 0.8
train = df[msk]
test = df[~msk]
encoding = MultiPointHeatmapEncoding(image_size, df, batch_size=64)

model_name = 'stacked_hourglass_tf2'
log_dir = "logs/{}".format(model_name)
model_filename = "saved-models/{}.h5".format(model_name)

train_gen = encoding.generator(train, batch_size)
test_gen = encoding.generator(test, batch_size, get_weights=True)

steps_per_epoch = len(train) // batch_size
validation_steps = len(test) // batch_size
if validation_steps == 0:
    validation_steps = 1
if steps_per_epoch == 0:
    steps_per_epoch = 1

cb_tensorboard = TensorBoard(log_dir=log_dir)
callback_save_images = CallbackHeatmapOutput(model, get_generator(test_gen), log_dir, encoding)
checkpoint = ModelCheckpoint(model_filename, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

history = model.fit_generator(
            get_generator(train_gen),
            validation_data=get_generator(test_gen),
            steps_per_epoch=steps_per_epoch,
            epochs=5000,
            validation_steps=validation_steps,
            verbose=2,
            use_multiprocessing=True,
            callbacks=[checkpoint, callback_save_images, cb_tensorboard]
        )