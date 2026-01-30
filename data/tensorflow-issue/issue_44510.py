from tensorflow.keras import layers
from tensorflow.keras import optimizers

import tensorflow as tf
from tensorflow import keras

import time
import socket

import os

tf.debugging.set_log_device_placement(True)

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
gpus = tf.config.list_physical_devices('GPU')
print(' gpus = ', gpus)

def get_compiled_model():
    # Make a simple 2-layer densely-connected neural network.
    inputs = keras.Input(shape=(784,))
    print("On GPU:0")
    with tf.device("/device:GPU:0"):
       x = keras.layers.Dense(256, activation="relu")(inputs)
       #assert x.device.endswith("/GPU:0")
       x = keras.layers.Dense(256, activation="relu")(inputs)
       x = keras.layers.Dense(256, activation="relu")(inputs)
       x = keras.layers.Dense(256, activation="relu")(inputs)
       x = keras.layers.Dense(256, activation="relu")(inputs)
       x = keras.layers.Dense(256, activation="relu")(inputs)
    print("On GPU:1")
    with tf.device("/device:GPU:1"): # Or GPU:1 for the 2nd GPU, GPU:2 for the 3rd etc.
       x = keras.layers.Dense(256, activation="relu")(inputs)
       #assert x.device.endswith("/GPU:1")
       x = keras.layers.Dense(256, activation="relu")(inputs)
       x = keras.layers.Dense(256, activation="relu")(inputs)
       x = keras.layers.Dense(256, activation="relu")(inputs)
       x = keras.layers.Dense(256, activation="relu")(inputs)
       x = keras.layers.Dense(256, activation="relu")(inputs)
    print("On GPU:2")
    with tf.device("/device:GPU:2"): # Or GPU:1 for the 2nd GPU, GPU:2 for the 3rd etc.
       x = keras.layers.Dense(256, activation="relu")(inputs)
       #assert x.device.endswith("/GPU:2")
       x = keras.layers.Dense(256, activation="relu")(inputs)
       x = keras.layers.Dense(256, activation="relu")(inputs)
       x = keras.layers.Dense(256, activation="relu")(inputs)
       x = keras.layers.Dense(256, activation="relu")(inputs)
       x = keras.layers.Dense(256, activation="relu")(inputs)
    print("On GPU:3")
    with tf.device("/device:GPU:3"): # Or GPU:1 for the 2nd GPU, GPU:2 for the 3rd etc.
      x = keras.layers.Dense(256, activation="relu")(x)
      #assert x.device.endswith("GPU:3")
      x = keras.layers.Dense(256, activation="relu")(x)
      x = keras.layers.Dense(256, activation="relu")(x)
      x = keras.layers.Dense(256, activation="relu")(x)
      x = keras.layers.Dense(256, activation="relu")(x)
      x = keras.layers.Dense(256, activation="relu")(x)
      outputs = keras.layers.Dense(10)(x)
    model = keras.Model(inputs, outputs)
 
    opt = keras.optimizers.Adam()
    model.compile(
        optimizer=opt,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
        experimental_run_tf_function=False,
    )
    return model


def get_dataset():
    batch_size = 32
    num_val_samples = 10000

    # Return the MNIST dataset in the form of a `tf.data.Dataset`.
    path = '/g/g92/jtaylor/workspace/TFnew/mnist.npz'
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data(path)

    # Preprocess the data (these are Numpy arrays)
    x_train = x_train.reshape(-1, 784).astype("float32") / 255
    x_test = x_test.reshape(-1, 784).astype("float32") / 255
    y_train = y_train.astype("float32")
    y_test = y_test.astype("float32")

    # Reserve num_val_samples samples for validation
    x_val = x_train[-num_val_samples:]
    y_val = y_train[-num_val_samples:]
    x_train = x_train[:-num_val_samples]
    y_train = y_train[:-num_val_samples]
    return (
        tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size),
        tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(batch_size),
        tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size),
    )

model = get_compiled_model()

# Train the model on all available devices.
train_dataset, val_dataset, test_dataset = get_dataset()
model.fit(train_dataset, epochs=2, validation_data=val_dataset)

# Test the model on all available devices.
model.evaluate(test_dataset)