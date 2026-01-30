from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf

from tensorflow.keras import datasets, layers, models
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

class NestedLayers(tf.keras.layers.Layer):
    def __init__(self):
        super(NestedLayers, self).__init__()
        self.units = [layers.Conv2D(16, (3,3), name="conv_2d_0"),
                      layers.Conv2D(16, (3,3), name="conv_2d_1"),
                      layers.Conv2D(16, (3,3), name="conv_2d_2")]

    def build(self, input_shape):
        for i in range(0,2):
            unit_input_shape = list(input_shape)
            unit_input_shape[-1] = 1
            unit = self.units[i]
            unit.build(unit_input_shape)

    def call(self, inputs):
        split_inputs = tf.split(value=inputs,
                                 num_or_size_splits=3,
                                 axis=-1,
                                 name="conv_grp_split")
        outputs = []
        for i in range(0,2):
            out = self.units[i](split_inputs[i])
            outputs.append(out)
        out = tf.keras.layers.concatenate(outputs, axis=-1, name="conv_grp_concat")
        return  out

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

model = models.Sequential()
model.add(layers.InputLayer(input_shape=(32, 32, 3)))
model.add(NestedLayers())
model.add(layers.Flatten())
model.add(layers.Dense(10, activation='softmax'))
model.summary()

import datetime
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

check_pt = tf.keras.callbacks.ModelCheckpoint(
                os.path.join(log_dir, "model.ckpt.{epoch:04d}-{val_loss:.06f}.hdf5"),
                monitor='val_loss',
                verbose=1,
                save_best_only=False,
                save_weights_only=True,
                mode='max',
                period=1)

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=2,
                    validation_data=(test_images, test_labels),
                    callbacks = [check_pt])

import tensorflow as tf

from tensorflow.keras import datasets, layers, models
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

class NestedLayers(tf.keras.layers.Layer):
    def __init__(self):
        super(NestedLayers, self).__init__()
        self.units = [layers.Conv2D(16, (3,3), name="conv_2d_0"),
                      layers.Conv2D(16, (3,3), name="conv_2d_1"),
                      layers.Conv2D(16, (3,3), name="conv_2d_2")]

    def build(self, input_shape):
        for i in range(0,2):
            unit_input_shape = list(input_shape)
            unit_input_shape[-1] = 1
            unit = self.units[i]
            unit.build(unit_input_shape)

    def call(self, inputs):
        split_inputs = tf.split(value=inputs,
                                 num_or_size_splits=3,
                                 axis=-1,
                                 name="conv_grp_split")
        outputs = []
        for i in range(0,2):
            out = self.units[i](split_inputs[i])
            outputs.append(out)
        out = tf.keras.layers.concatenate(outputs, axis=-1, name="conv_grp_concat")
        return  out

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

model = models.Sequential()
model.add(layers.InputLayer(input_shape=(32, 32, 3)))
model.add(NestedLayers())
model.add(layers.Flatten())
model.add(layers.Dense(10, activation='softmax'))
model.summary()

import datetime

check_pt = tf.keras.callbacks.ModelCheckpoint(
                os.path.join("model.ckpt.{epoch:04d}-{val_loss:.06f}.hdf5"),
                monitor='val_loss',
                verbose=1,
                save_best_only=False,
                save_weights_only=True,
                mode='max',
                period=1)

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=2,
                    validation_data=(test_images, test_labels),
                    callbacks = [check_pt])