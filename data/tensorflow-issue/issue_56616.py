from tensorflow.keras import layers

import tensorflow as ts
import numpy as np

from tensorflow.keras import Sequential as Sq
from tensorflow.keras.layers import Dense as Den
from tensorflow.keras.layers import MaxPooling2D as Max
from tensorflow.keras.layers import Conv2D as conv

# Entry point of python programme
from callbacks.TrainingEpochCallback import EpochCallBack


def program_entry():
    # Get inbuilt dataset
    data = ts.keras.datasets.fashion_mnist

    # Tuples of training data sets by calling load_data() method
    (training_images, training_labels), (test_images, test_labels) = data.load_data()

    # Create a new array, of 'training_images' pixel representation, each pixel range is between 0 - 255, so divide
    # Each of them array element with 255, should make it under zero and one, this is called normalizing the image
    # it will help ML model enhance performance
    training_images = training_images / 255.0

    # Create a new array, of 'training_images' pixel representation, each pixel range is between 0 - 255, so divide
    # Each of them array element with 255, should make it under zero and one, this is called normalizing the image
    # it will help ML model enhance performance
    test_images = test_images / 255.0

    # Create a Sequential model instance with a simple Neural Network Layer
    # That contains 128 layer and 128 neuron
    l0 = [
        conv(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        Max(pool_size=(2, 2)),
        conv(64, (3, 3), activation='relu'),
        Max(pool_size=(2, 2)),
        ts.keras.layers.Flatten(),
        Den(128, activation='relu'),
        Den(10, activation='softmax')
    ]

    model = Sq(l0)

    # Configures the model for training
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Trains the model for a fixed number of epochs (iterations on a dataset)
    model.fit(training_images, training_labels, epochs=50)

    # Evaluate the model with test data set
    model.evaluate(test_images, test_labels)

    # Make prediction on test data set
    classifications = model.predict(test_images)
    print(classifications[0])
    print(test_labels[0])


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Entry point to program
    program_entry()