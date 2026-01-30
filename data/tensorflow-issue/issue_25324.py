import random
from tensorflow.keras import layers
from tensorflow.keras import models

#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
import sys

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model


def fit_keras_model():
    data = np.random.random((1000, 32))
    labels = np.random.random((1000, 10))
    # source https://keras.io/getting-started/functional-api-guide/
    # This returns a tensor
    inputs = Input(shape=(32,))

    # a layer instance is callable on a tensor, and returns a tensor
    x = Dense(64, activation='relu')(inputs)
    x = Dense(64, activation='relu')(x)
    predictions = Dense(10, activation='softmax')(x)

    # This creates a model that includes
    # the Input layer and three Dense layers
    model = Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(data, labels)  # starts training
    return model


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'eager':
        tf.enable_eager_execution()  # fails with eager execution enabled
    fit_keras_model()
    print('success')