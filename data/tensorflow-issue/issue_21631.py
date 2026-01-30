from tensorflow.keras import layers
from tensorflow.keras import models

from __future__ import division, print_function, unicode_literals

import logging
logging.basicConfig(level=logging.WARNING)

import tensorflow as tf
import tensorflow.contrib.eager as tfe
#tf.enable_eager_execution() # same problem in eager mode or graph mode
from tensorflow import keras
import numpy as np

n_epochs = 5
batch_size = 32

def create_keras_model():
    keras.backend.clear_session()
    model = keras.models.Sequential([
        keras.layers.Dense(300, activation="relu", input_shape=[28*28]),
        keras.layers.Dense(100, activation="relu"),
        keras.layers.Dense(10, activation="softmax")
    ])
    optimizer = tf.train.MomentumOptimizer(learning_rate=0.01, momentum=0.9)
    model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer,
                  metrics=["accuracy"])
    return model

def create_keras_estimator():
    model = create_keras_model()
    input_name = model.input_names[0]
    estimator = tf.keras.estimator.model_to_estimator(model)
    return estimator, input_name

def create_dnn_estimator():
    input_name = "pixels"
    pixels = tf.feature_column.numeric_column(input_name, shape=[28 * 28])
    estimator = tf.estimator.DNNClassifier(hidden_units=[300, 100, 10],
                                           n_classes=10,
                                           feature_columns=[pixels])
    return estimator, input_name

def train_and_evaluate_estimator(estimator, input_name,
                                 X_train, y_train, X_test, y_test):
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={input_name: X_train}, y=y_train, num_epochs=5, batch_size=32,
        shuffle=True)
    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={input_name: X_test}, y=y_test, shuffle=False)
    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn)
    eval_spec = tf.estimator.EvalSpec(input_fn=test_input_fn)
    return tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

def load_and_scale_MNIST():
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    X_train = X_train.astype(np.float32).reshape(-1, 28*28) / 255.0
    X_test = X_test.astype(np.float32).reshape(-1, 28*28) / 255.0
    y_train = y_train.astype(np.int32)
    y_test = y_test.astype(np.int32)
    return X_train, y_train, X_test, y_test

if __name__ == '__main__':
    X_train, y_train, X_test, y_test = load_and_scale_MNIST()
    
    for run in range(2):
        if run == 0:
            print("==== Training with 1D labels... ====")
            assert y_train.ndim == 1 and y_test.ndim == 1
        else:
            y_train = y_train.reshape(-1, 1)
            y_test = y_test.reshape(-1, 1)
            print()
            print("==== Training with 2D labels... ====")
            assert y_train.ndim == 2 and y_test.ndim == 2

        print("Keras model:")
        keras_model = create_keras_model()
        keras_model.fit(X_train, y_train, epochs=n_epochs,
                        batch_size=batch_size, verbose=0)
        print("    Eval:", keras_model.evaluate(X_test, y_test, verbose=0))

        print("Keras model converted to an estimator:")
        keras_estimator, input_name = create_keras_estimator()
        print("    Eval:", train_and_evaluate_estimator(
            keras_estimator, input_name, X_train, y_train, X_test, y_test))

        print("DNNClassifier:")
        dnn_estimator, input_name = create_dnn_estimator()
        print("    Eval:", train_and_evaluate_estimator(
            dnn_estimator, input_name, X_train, y_train, X_test, y_test))