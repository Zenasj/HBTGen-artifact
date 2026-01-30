from tensorflow.keras import layers
from tensorflow.keras import optimizers

"""
Reproducer for #48057

Run on single GPU OK. Run on two GPUs and it fails during evaluation.
"""
import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image_dataset_from_directory


INPUT_SHAPE = (224, 224, 3)
IMAGE_SIZE = (224, 224)
NUM_CLASSES = 5
EPOCHS = 1


def load_img_datasets(folder, image_size, batch_size=32):
    training_dataset = image_dataset_from_directory(os.path.join(folder,
                                                                 "train"),
                                                    labels="inferred",
                                                    label_mode="categorical",
                                                    batch_size=batch_size,
                                                    image_size=image_size)
    validation_dataset = image_dataset_from_directory(os.path.join(folder,
                                                                   "val"),
                                                      labels="inferred",
                                                      label_mode="categorical",
                                                      batch_size=batch_size,
                                                      image_size=image_size)
    testing_dataset = image_dataset_from_directory(os.path.join(folder,
                                                                "test"),
                                                   labels="inferred",
                                                   label_mode="categorical",
                                                   batch_size=batch_size,
                                                   image_size=image_size)
    return training_dataset, validation_dataset, testing_dataset


def create_model():
    inputs = keras.Input(shape=INPUT_SHAPE)
    x = tf.cast(inputs, tf.float32)
    x = keras.layers.Conv2D(1, (2, 2), strides=(1, 1), padding='same')(x)
    x = keras.layers.Lambda(tf.nn.local_response_normalization)(x)
    # if I use x = keras.layers.BatchNormalization()(x) then OK
    x = keras.layers.Activation('relu')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')(x)
    return keras.Model(inputs=inputs, outputs=x, name="toy")


if __name__ == '__main__':
    training_data, validation_data, testing_data = \
        load_img_datasets("./data/lms/224_split", IMAGE_SIZE)
    # These ^ are tensorflow.python.data.ops.dataset_ops.BatchDataset

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = create_model()
        optimizer = tf.keras.optimizers.Adam()
        metrics = ['accuracy']
        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizer,
                      metrics=metrics)

    history = model.fit(training_data, epochs=EPOCHS,
                        validation_data=validation_data)

    loss, accuracy = model.evaluate(testing_data)
    # Crashes here ^