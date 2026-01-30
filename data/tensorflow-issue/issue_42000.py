import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

"""
Note: when add "@tf.function" in front from train_step, report error; else, successfully run two times.
Environment; tensorflow2.x python3.x
author: masterqkk
"""


def build_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(10, input_shape=[2]))
    model.add(tf.keras.layers.Dense(1))
    return model

def compute_loss(batch_true, batch_pred):
    losses = tf.losses.mean_squared_error(batch_true, batch_pred)
    loss = tf.reduce_mean(losses)
    return loss


@tf.function
def train_step(model, batch_input, batch_label, optimizer):
    with tf.GradientTape() as tape:
        preds = model(batch_input)
        loss = compute_loss(batch_label, preds)
    trainable_variables = model.trainable_variables
    grads = tape.gradient(loss, trainable_variables)
    optimizer.apply_gradients(grads_and_vars=zip(grads, trainable_variables))
    return loss


def train_epoch(model, data_batchs, optimizer):
    for (batch_input, batch_label) in data_batchs:
        loss = train_step(model, batch_input, batch_label, optimizer)


def train_model(model, data_batchs, optimizer):
    for i in np.arange(1):
        train_epoch(model, data_batchs, optimizer)


def load_data():
    np.random.seed(0)
    x = np.array(np.random.random(size=(10, 2)), np.float32)
    y = np.array(np.random.random(size=(10, 1)), np.float32)
    data = tf.data.Dataset.from_tensor_slices((x, y))
    data_batchs = data.batch(batch_size=5)
    return data_batchs


def run_flow():
    data_batchs = load_data()
    model = build_model()
    optimizer = tf.keras.optimizers.Adam()
    train_model(model, data_batchs, optimizer)
    tf.print('model train finished.')


for i in np.arange(2):
    run_flow()

# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

"""
Note: when add "@tf.function" in front from train_step, report error; else, successfully run two times.
Environment; tensorflow2.x python3.x
author: masterqkk
"""


def build_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(10, input_shape=[2]))
    model.add(tf.keras.layers.Dense(1))
    return model

def compute_loss(batch_true, batch_pred):
    losses = tf.losses.mean_squared_error(batch_true, batch_pred)
    loss = tf.reduce_mean(losses)
    return loss

def get_apply_grad_fn():
    @tf.function
    def train_step(model, batch_input, batch_label, optimizer):
        with tf.GradientTape() as tape:
            preds = model(batch_input)
            loss = compute_loss(batch_label, preds)
        trainable_variables = model.trainable_variables
        grads = tape.gradient(loss, trainable_variables)
        optimizer.apply_gradients(grads_and_vars=zip(grads, trainable_variables))
        return loss
    return train_step


def train_epoch(model, data_batchs, optimizer):
    for (batch_input, batch_label) in data_batchs:
        model_apply_grads = get_apply_grad_fn()
        loss = model_apply_grads(model, batch_input, batch_label, optimizer)


def train_model(model, data_batchs, optimizer):
    for i in np.arange(1):
        train_epoch(model, data_batchs, optimizer)


def load_data():
    np.random.seed(0)
    x = np.array(np.random.random(size=(10, 2)), np.float32)
    y = np.array(np.random.random(size=(10, 1)), np.float32)
    data = tf.data.Dataset.from_tensor_slices((x, y))
    data_batchs = data.batch(batch_size=5)
    return data_batchs


def run_flow():
    data_batchs = load_data()
    model = build_model()
    optimizer = tf.keras.optimizers.Adam()
    model_apply_grads = get_apply_grad_fn()
    train_model(model, data_batchs, optimizer)
    tf.print('model train finished.')


for i in np.arange(2):
    run_flow()

for i in np.arange(10):
       run_flow(random_seed=i)

def build_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(10, input_shape=[2]))
    model.add(tf.keras.layers.Dense(1))
    return model

def compute_loss(batch_true, batch_pred):
    losses = tf.losses.mean_squared_error(batch_true, batch_pred)
    loss = tf.reduce_mean(losses)
    return loss

def train_step(model, batch_input, batch_label, optimizer):
    with tf.GradientTape() as tape:
        preds = model(batch_input)
        loss = compute_loss(batch_label, preds)
    trainable_variables = model.trainable_variables
    grads = tape.gradient(loss, trainable_variables)
    optimizer.apply_gradients(grads_and_vars=zip(grads, trainable_variables))
    return loss

def train_model(model, data_batchs, optimizer):
    # Create a new tf.function from train_step that will be used in this experiment only
    train_step_function = tf.function(train_step)
    num_epochs = 10
    for i in np.arange(num_epochs):
        for (batch_input, batch_label) in data_batchs:
            loss = train_step_function(model, batch_input, batch_label, optimizer)

def load_data():
    np.random.seed(0)
    x = np.array(np.random.random(size=(10, 2)), np.float32)
    y = np.array(np.random.random(size=(10, 1)), np.float32)
    data = tf.data.Dataset.from_tensor_slices((x, y))
    data_batchs = data.batch(batch_size=5)
    return data_batchs

def run_flow():
    data_batchs = load_data()
    model = build_model()
    optimizer = tf.keras.optimizers.Adam()
    train_model(model, data_batchs, optimizer)

for i in np.arange(10):
    print("Running experiment {}".format(i))
    run_flow()