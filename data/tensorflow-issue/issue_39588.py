import random
from tensorflow.keras import layers

py
import typing

import numpy as np
from tensorflow import keras
import tensorflow_datasets as tfds


def build_neural_network(input_dimension: int, number_of_classes: int, compile_options: dict):
    model = keras.Sequential()
    model.add(keras.layers.Dense(112, activation='relu', input_dim=input_dimension))
    model.add(keras.layers.Dense(112, activation='relu'))
    model.add(keras.layers.Dense(number_of_classes, activation='softmax'))

    model.compile(**compile_options)

    print(model.summary())

    return model

def load_in_images_and_labels_and_reshape(dataset) -> typing.Tuple[np.ndarray, np.ndarray]:
    images = []
    labels = []
    for image, label in tfds.as_numpy(dataset):
        new_image_shape = image.shape[0] * image.shape[1]
        images.append(image.reshape(new_image_shape))
        labels.append(label)

    return np.array(images), np.array(labels)


def train_neural_network(is_random_weighing: bool):
    dataset_train      = tfds.load('emnist', split='train', as_supervised=True)
    dataset_validation = tfds.load('emnist', split='test', as_supervised=True)

    train_images, train_labels           = load_in_images_and_labels_and_reshape(dataset_train)
    validation_images, validation_labels = load_in_images_and_labels_and_reshape(dataset_validation)
    train_labels      = keras.utils.to_categorical(train_labels)
    validation_labels = keras.utils.to_categorical(validation_labels)

    print("load")
    compile_options =  {
        "loss": "categorical_crossentropy",
        "optimizer": "adam",
        "metrics": ["categorical_accuracy"],
        "weighted_metrics": ["categorical_accuracy"]
    }
    network = build_neural_network(train_images.shape[-1], len(train_labels[0]), compile_options)

    fit_options = {    
        "batch_size": 2048,
        "epochs": 10,
        "verbose": 1,
        "workers": 1
    }
    if is_random_weighing:
        random_weights = np.random.rand(len(validation_images))
        validation_data_tuple = (validation_images, validation_labels, random_weights)
    else:
        validation_data_tuple = (validation_images, validation_labels)
    history = network.fit(train_images, train_labels, validation_data=validation_data_tuple, **fit_options)


if __name__ == "__main__":
    is_random_weighing = True
    train_neural_network(is_random_weighing)