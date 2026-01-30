from tensorflow.keras import layers

element [[[x1, y1, z1], [c1,]], [[x2, y2, z2], [c2,]], [[x3, y3, z3], [c3,]]] 
element [[[x1, y1, z1], [c1,]], [[x2, y2, z2], [c2,]], [[x3, y3, z3], [c3,]]] 
...

# %%
import os

import tensorflow as tf # tensorflow nightly, version>=2.5
from tensorflow import keras
from tensorflow.image import crop_to_bounding_box as tfimgcrop
from tensorflow.keras.preprocessing import image_dataset_from_directory

BATCH_SIZE=32 # Adjust?

IMG_SIZE=(224, 224)
IMG_SHAPE = IMG_SIZE + (3,)

# %%
_URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
path_to_zip = tf.keras.utils.get_file('cats_and_dogs.zip', origin=_URL, extract=True)
PATH = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered')

train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')

train_dataset = tf.keras.preprocessing.image_dataset_from_directory(train_dir,
                                             shuffle=False,
                                             label_mode='categorical',
                                             batch_size=32,
                                             image_size=IMG_SIZE)
validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(validation_dir,
                                             shuffle=False,
                                             label_mode='categorical',
                                             batch_size=32,
                                             image_size=IMG_SIZE)

# %%
base_model1 = tf.keras.applications.MobileNetV3Large(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet',
                                               minimalistic=False,
                                               pooling=max,
                                               dropout_rate=0.2)
base_model2 = tf.keras.applications.MobileNetV3Large(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet',
                                               minimalistic=False,
                                               pooling=max,
                                               dropout_rate=0.2)
base_model3 = tf.keras.applications.MobileNetV3Large(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet',
                                               minimalistic=False,
                                               pooling=max,
                                               dropout_rate=0.2)

# %%
pre_concat_layer1 = tf.keras.layers.Dense(64, 
                                        activation='relu', 
                                        kernel_initializer='random_uniform', 
                                        bias_initializer='zeros')
pre_concat_layer2 = tf.keras.layers.Dense(64, 
                                        activation='relu', 
                                        kernel_initializer='random_uniform', 
                                        bias_initializer='zeros')
pre_concat_layer3 = tf.keras.layers.Dense(64, 
                                        activation='relu', 
                                        kernel_initializer='random_uniform', 
                                        bias_initializer='zeros')

post_concat_layer = tf.keras.layers.Dense(128, 
                                        activation='relu', 
                                        kernel_initializer='random_uniform', 
                                        bias_initializer='zeros')
prediction_layer = tf.keras.layers.Dense(2, 
                                        activation='softmax', 
                                        kernel_initializer='random_uniform', 
                                        bias_initializer='zeros')

# %%
input1 = tf.keras.Input(shape=(64, 64, 3), name="First")
input2 = tf.keras.Input(shape=(64, 64, 3), name="Second")
input3 = tf.keras.Input(shape=(64, 64, 3), name="Third")

x = base_model1(input1, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.BatchNormalization()(x)
x = pre_concat_layer1(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.BatchNormalization()(x)
body1 = tf.keras.Model(input1, outputs)

x = base_model2(input2, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.BatchNormalization()(x)
x = pre_concat_layer2(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.BatchNormalization()(x)
body2 = tf.keras.Model(input2, outputs)

x = base_model3(input3, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.BatchNormalization()(x)
x = pre_concat_layer3(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.BatchNormalization()(x)
body3 = tf.keras.Model(input3, outputs)

# %%
body1.get_layer("MobilenetV3large")._name = "MobilenetV3large1"
body2.get_layer("MobilenetV3large")._name = "MobilenetV3large2"
body3.get_layer("MobilenetV3large")._name = "MobilenetV3large3"

# %%
combinedInput = tf.keras.layers.concatenate([body1.output, body2.output, body3.output])
x = post_concat_layer(combinedInput)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.BatchNormalization()(x)
foutput = prediction_layer(x)
final_model = tf.keras.Model(inputs=[body1.input, body2.input, body3.input], outputs=foutput)

# %%
def resize_data1(images, classes):
    return (tfimgcrop(images,
                        offset_height=0,
                        offset_width=0,
                        target_height=64,
                        target_width=64),
                    classes)
def resize_data2(images, classes):
    return (tfimgcrop(images,
                        offset_height=0,
                        offset_width=64,
                        target_height=64,
                        target_width=64),
                    classes)
def resize_data3(images, classes):
    return (tfimgcrop(images,
                        offset_height=0,
                        offset_width=128,
                        target_height=64,
                        target_width=64),
                    classes)

# %%
train_dataset_unb = train_dataset.unbatch()
train_dataset1 = train_dataset_unb.map(resize_data1)
train_dataset2 = train_dataset_unb.map(resize_data2)
train_dataset3 = train_dataset_unb.map(resize_data3)
train_dataset_zip = tf.data.Dataset.zip((train_dataset1, train_dataset2, train_dataset3))

validation_dataset_unb = validation_dataset.unbatch()
validation_dataset1 = validation_dataset_unb.map(resize_data1)
validation_dataset2 = validation_dataset_unb.map(resize_data2)
validation_dataset3 = validation_dataset_unb.map(resize_data3)
validation_dataset_zip = tf.data.Dataset.zip((validation_dataset1, validation_dataset2, validation_dataset3))

# %%
final_model.compile()

# %%
history = final_model.fit(train_dataset_zip,
                        epochs=999, 
                        validation_data=validation_dataset_zip,
                        validation_steps=32
                        )