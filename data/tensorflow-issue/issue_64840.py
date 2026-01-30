import random
from tensorflow.keras import layers
from tensorflow.keras import models

# Training a neural network on Fashion MNIST using ℓ2 Regularization

# Importing essential package
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# Loading Fashion MNIST dataset and preprocessing
(X_train_full, y_train_full), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
# Scaling pixel values to the range [0, 1]
X_train_full = X_train_full / 255.0
X_test = X_test / 255.0
# Splitting training set into validation and training subsets
X_valid, X_train = X_train_full[:5000], X_train_full[5000:]
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
# Calculating mean and standard deviation of training pixel values
pixel_means = X_train.mean(axis=0, keepdims=True)
pixel_stds = X_train.std(axis=0, keepdims=True)
# Standardizing the input features using mean and standard deviation
X_train_scaled = (X_train - pixel_means) / pixel_stds
X_valid_scaled = (X_valid - pixel_means) / pixel_stds
X_test_scaled = (X_test - pixel_means) / pixel_stds

# Setting random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Creating a Sequential model
model = keras.models.Sequential([
    # Input layer: Flatten layer to convert 2D input (28x28) to 1D array
    keras.layers.Flatten(input_shape=[28, 28]),
    # Hidden layer 1: Dense layer with 300 neurons, ELU activation function,
    # He initialization, and ℓ2 regularization with a regularization strength of 0.01
    keras.layers.Dense(300, activation="elu",
                       kernel_initializer="he_normal",
                       kernel_regularizer=keras.regularizers.l2(0.01)),
    # Hidden layer 2: Dense layer with 100 neurons, ELU activation function,
    # He initialization, and ℓ2 regularization with a regularization strength of 0.01
    keras.layers.Dense(100, activation="elu",
                       kernel_initializer="he_normal",
                       kernel_regularizer=keras.regularizers.l2(0.01)),
    # Output layer: Dense layer with 10 neurons (for 10 classes), softmax activation function,
    # and ℓ2 regularization with a regularization strength of 0.01
    keras.layers.Dense(10, activation="softmax",
                       kernel_regularizer=keras.regularizers.l2(0.01))
])

# Compiling the model
model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam", metrics=["accuracy"])

# Defining number of epochs
n_epochs = 2

# Training the model
history = model.fit(X_train_scaled, y_train, epochs=n_epochs,
                    validation_data=(X_valid_scaled, y_valid))