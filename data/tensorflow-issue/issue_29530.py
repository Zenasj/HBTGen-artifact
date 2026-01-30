from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

# %%
import tensorflow as tf
import json
from ruamel import yaml


# %%
def mini_cnn(input_shape, num_classes):
    inputs = tf.keras.layers.Input(shape=input_shape)

    x = tf.keras.layers.Conv2D(16, (3, 3), padding="same", use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization(axis=-1)(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=2)(x)
    x = tf.keras.layers.Conv2D(32, (3, 3), padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(axis=-1)(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=2)(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(axis=-1)(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=2)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(64, use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(axis=-1)(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Dropout(0.25)(x)

    logits = tf.keras.layers.Dense(num_classes, use_bias=True, name="logits")(x)
    probas = tf.keras.layers.Activation(activation="softmax", name="probas")(logits)

    model = tf.keras.Model(inputs=inputs, outputs=probas, name="model")

    return model


# %%
m = mini_cnn((64, 64, 3), 2)

## %%
try:
    c = m.get_config()
    m2 = tf.keras.models.model_from_config(c)
except:
    print("c error")

try:
    y = m.to_yaml()
    m2 = tf.keras.models.model_from_yaml(y)
except:
    print("y error")

try:
    j = m.to_json()
    m2 = tf.keras.models.model_from_json(j)
except:
    print("j error")

# %%
import traceback

import tensorflow as tf


# %%
def mini_cnn(input_shape, num_classes):
    inputs = tf.keras.layers.Input(shape=input_shape)

    x = tf.keras.layers.Conv2D(16, (3, 3), padding="same", use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization(axis=-1)(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=2)(x)
    x = tf.keras.layers.Conv2D(32, (3, 3), padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(axis=-1)(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=2)(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(axis=-1)(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=2)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(64, use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(axis=-1)(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Dropout(0.25)(x)

    logits = tf.keras.layers.Dense(num_classes, use_bias=True, name="logits")(x)
    probas = tf.keras.layers.Activation(activation="softmax", name="probas")(logits)

    model = tf.keras.Model(inputs=inputs, outputs=probas, name="model")

    return model


def mini_cnn_with_lambda(input_shape, num_classes):
    inputs = tf.keras.layers.Input(shape=input_shape)

    x = tf.keras.layers.Conv2D(16, (3, 3), padding="same", use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization(axis=-1)(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=2)(x)
    x = tf.keras.layers.Conv2D(32, (3, 3), padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(axis=-1)(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=2)(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(axis=-1)(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=2)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(64, use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(axis=-1)(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    x = tf.keras.layers.Lambda(lambda x: 2. * x)(x)

    logits = tf.keras.layers.Dense(num_classes, use_bias=True, name="logits")(x)
    probas = tf.keras.layers.Activation(activation="softmax", name="probas")(logits)

    model = tf.keras.Model(inputs=inputs, outputs=probas, name="model")

    return model


# %%
m = mini_cnn_with_lambda((64, 64, 3), 2)

## %%
try:
    c = m.get_config()
    m2 = tf.keras.models.model_from_config(c)
    error_c = "Success on dict loading"
except:
    error_c = "Error on loading model from config\n"
    error_c += traceback.format_exc()

try:
    y = m.to_yaml()
    m2 = tf.keras.models.model_from_yaml(y)
    error_y = "Success on YAML loading"
except:
    error_y = "Error on loading model from yaml\n"
    error_y += traceback.format_exc()

try:
    j = m.to_json()
    m2 = tf.keras.models.model_from_json(j)
    error_j = "Success on json loading"
except:
    error_j = "Error on loading model from json\n"
    error_j += traceback.format_exc()

# %%

print("-- Status & traceback from config")
print(error_c)

print("-- Status & traceback from yaml")
print(error_y)

print("-- Status & traceback from json")
print(error_j)

c = m.get_config()
m2 = tf.keras.models.model_from_config(c)