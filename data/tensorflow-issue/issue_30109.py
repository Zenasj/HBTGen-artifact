from tensorflow.keras import layers
from tensorflow.keras import models

from tensorflow import keras
import tensorflow as tf
import numpy as np
import random

time_steps = 5
sample_width = 20
kernel_size = 3
num_filters = 5
num_classes = 5

def build_graph():
    with tf.name_scope("SequenceProcess"):
        sequence_input = keras.layers.Input(shape=(time_steps, sample_width))

        with tf.name_scope("TimeDistributed"):
            sample_input = keras.layers.Input(shape=(sample_width,))

            conv = keras.layers.Reshape([sample_width, 1])(sample_input)
            conv = keras.layers.BatchNormalization(momentum=0.01)(conv)
            conv = keras.layers.Conv1D(num_filters, kernel_size, padding='same')(conv)
            encoded_sample = keras.layers.Reshape((num_filters * sample_width,))(conv)

            sample_model = keras.models.Model(sample_input, encoded_sample)

        processed_samples = keras.layers.TimeDistributed(sample_model)(sequence_input)
        fc = keras.layers.Dense(num_classes, activation='softmax')(processed_samples)

        sequence_model = keras.models.Model(sequence_input, fc)
        sequence_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['sparse_categorical_accuracy'])

    return sequence_model

def generate_data():
    steps = []
    for i in range(time_steps):
        sample_data = np.random.rand(sample_width)
        steps.append(sample_data)
    steps = np.array(steps)

    labels = []
    for i in range(time_steps):
        label = random.randint(0, num_classes - 1)
        labels.append([label])
    labels = np.array(labels)

    input_data = []
    input_labels = []

    for i in range(1000):
        input_data.append(steps)
        input_labels.append(labels)

    return np.array(input_data), np.array(input_labels)

if __name__ == '__main__':
    data, labels = generate_data()
    model = build_graph()
    model.fit(data, labels, 10, 100, validation_split= 0.5)