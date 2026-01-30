from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras import optimizers

base_model = tf.keras.applications.ResNet50(include_top=False, weights='imagenet')
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
output_layer = tf.keras.layers.Dense(1)

model = tf.keras.Sequential([
    base_model,
    global_average_layer,
    output_layer
])


model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_dataset,
                    epochs=epochs,
                    validation_data=val_dataset)

loss, accuracy = model.evaluate(val_dataset)

# Save the model
model.save('model-v1.keras')

model.summary()

n_model = tf.keras.models.load_model('model-v1.keras')

n_model.summary()

input_shape = (224, 224, 3)
model = tf.keras.models.load_model('model.keras', compile=False, custom_objects={'input_shape': input_shape})

base_model = keras.applications.MobileNetV2(input_shape=(224,224,3),include_top=False,weights="imagenet")
model = keras.Sequential([
    base_model,
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dropout(rate=0.5),
    keras.layers.Dense(8,activation='softmax'),
])

model.compile(...)
model.fit(...)

model.summary() #OK

model.save("test_mobilenetv2_save.keras")

model_2 = keras.saving.load_model("test_mobilenetv2_save.keras") #Error
model_2 = keras.saving.load_model("test_mobilenetv2_save.keras",compile=False, custom_objects={'input_shape': input_shape}) #Error

model_2.summary()

base_model2 = keras.applications.MobileNetV2(input_shape=(224,224,3),include_top=False,weights="imagenet")

base_model2.trainable = True

model2 = keras.Sequential([
    base_model2,
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dropout(rate=0.5),
    keras.layers.Dense(8,activation='softmax'),
])

model2.compile(
    loss=loss,
    # optimizer=keras.optimizers.SGD(learning_rate=0.01),
    optimizer = optimizer,
    metrics=[
        metrics.CategoricalAccuracy(),
        metrics.TopKCategoricalAccuracy(k=3),
    ]
)

model2.summary()

model2.predict(np.ones((32,224,224,3))) #must build model first or you not able to load weights

model2.load_weights("mobilenetv2_weights_only.weights.h5", skip_mismatch=False)

model = build_model()
model.build(input_shape=(None, *INPUT_SHAPE))
model.load_weights('model.weights.h5')

import pathlib
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

'''
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
'''
format_1 = '/kaggle/input/flower-dataset/train'
format_2 = '/kaggle/input/flower-dataset/valid'
train_dir = pathlib.Path(format_1)
valid_dir = pathlib.Path(format_2)
train_count = len(list(train_dir.glob('*/*.jpg')))
valid_count = len(list(valid_dir.glob('*/*.jpg')))

print(train_count)
print(valid_count)

CLASS_NAMES = np.array([item.name for item in train_dir.glob('*') if item.is_dir()])

image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
BATCH_SIZE = 32
IMG_HEIGHT = 224
IMG_WIDTH = 224
train_data_gen = image_generator.flow_from_directory(directory=str(train_dir),
                                                     batch_size=BATCH_SIZE,
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                     shuffle=True,
                                                     classes=list(CLASS_NAMES))
valid_data_gen = image_generator.flow_from_directory(directory=str(valid_dir),
                                                     batch_size=BATCH_SIZE,
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                     shuffle=True,
                                                     classes=list(CLASS_NAMES))
print(CLASS_NAMES)
ResNet50 = tf.keras.applications.resnet_v2.ResNet50V2(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
ResNet50.trainable = True
my_net = tf.keras.models.Sequential()
my_net.add(ResNet50)
my_net.add(tf.keras.layers.GlobalAveragePooling2D())
my_net.add(tf.keras.layers.Dense(100, activation='softmax'))
my_net.summary()
my_net.compile(optimizer=tf.keras.optimizers.Adamax(0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy'])
epoch_steps = train_count // BATCH_SIZE
val_steps = valid_count // BATCH_SIZE
my_net.fit_generator(
    train_data_gen,
    steps_per_epoch=epoch_steps,
    epochs=5,
    validation_data=valid_data_gen,
    validation_steps=val_steps
)
my_net.save('flower.h5')

import tensorflow as tf
from tensorflow.keras.applications import EfficientNetV2S

V2S_model = EfficientNetV2S(weights='imagenet',
                        include_top=False,
                        input_shape=(224, 224, 3))

for layer in V2S_model.layers:
  layer.trainable = False
from tensorflow.keras import layers

image_preprocess = tf.keras.Sequential([
    tf.keras.Input((None,None,3)),
    layers.Resizing(224,224, crop_to_aspect_ratio = True),
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2)
], name = "image_aug")

transfer_model = tf.keras.Sequential([
    tf.keras.Input((None,None,3)),
    image_preprocess,
    V2S_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(1, activation = 'sigmoid')
])

transfer_model.summary()

metrics = [tf.keras.metrics.BinaryAccuracy(threshold=0.5, name='accuracy'),
           tf.keras.metrics.AUC(),
           tf.keras.metrics.TruePositives(),
           tf.keras.metrics.TrueNegatives(),
           tf.keras.metrics.FalsePositives(),
           tf.keras.metrics.FalseNegatives()
          ]

transfer_model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                       optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                       metrics=metrics)

BATCH_SIZE = 16
IMAGE_SIZE = (224,224)
SEED = 42

# This sets up a training and validation set from our ../data/ directory
train_dataset = tf.keras.utils.image_dataset_from_directory(
    '../data/',
    class_names = ['not_niblet','niblet'],
    color_mode='rgb',
    batch_size=BATCH_SIZE,
    image_size=IMAGE_SIZE,
    shuffle=True,
    validation_split=0.2,
    subset='training',
    seed=SEED)

# This is the validation set. Notice `shuffle = FALSE` and `subset = validation`
val_dataset = tf.keras.utils.image_dataset_from_directory(
    '../data/',
    class_names = ['not_niblet','niblet'],
    color_mode='rgb',
    batch_size=BATCH_SIZE,
    image_size=IMAGE_SIZE,
    shuffle=True,
    validation_split=0.2,
    subset='validation',
    seed=SEED)

from tensorflow.keras.callbacks import EarlyStopping

es = EarlyStopping(patience=15, monitor='val_loss')

history = transfer_model.fit(train_dataset, epochs=100,
                             validation_data=val_dataset,
                             callbacks=[es])

transfer_model.save('../models/transfer_model_gt_2024_04_23.keras')

transfer_model_loaded = tf.keras.models.load_model('../models/transfer_model_gt_2024_04_23.keras')