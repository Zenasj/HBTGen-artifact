from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense

print(tf.__version__)


def train_model(
        training_features,
        prediction_features
):
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=([len(training_features)])))
    model.add(Dense(len(prediction_features), activation='sigmoid'))

    model.summary()

    tf.contrib.saved_model.save_keras_model(model, "./saved_models", serving_only=True)