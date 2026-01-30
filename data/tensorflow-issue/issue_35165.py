from tensorflow.keras import layers
from tensorflow.keras import optimizers

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import time
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class MyModel(keras.Model):

    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = keras.layers.Conv2D(64, [3, 3])
        self.conv2 = keras.layers.Conv2D(64, [3, 3])
        self.flatten = keras.layers.Flatten()

    def call(self, inputs, training):
        images = inputs[0]
        targets = inputs[1]
        x = self.conv1(images)
        x = self.conv2(x)
        x = self.flatten(x)
        loss = tf.reduce_mean(x, axis=1) - tf.reduce_mean(targets, axis=1)
        return loss

def create_dataset():
    X = np.zeros([10, 224, 224, 3], dtype=np.float32)
    Y = np.zeros([10, 1000], dtype=np.float32)
    x_ds = tf.data.Dataset.from_tensor_slices(X)
    y_ds = tf.data.Dataset.from_tensor_slices(Y)
    ds = tf.data.Dataset.zip((x_ds, y_ds))
    ds = ds.batch(32)
    return ds

@tf.function
def train_one_step(model, optim, inputs):
    with tf.GradientTape() as tape:
        loss = model(inputs, training=True)
        loss = tf.reduce_mean(loss)
    grads = tape.gradient(loss, model.trainable_variables)
    grads_and_vars = zip(grads, model.trainable_variables)
    optim.apply_gradients(grads_and_vars)
    return loss


@tf.function
def in_graph_training_loop(model, optim, dataset):
    step = 0
    for inputs in dataset:
        loss = train_one_step(model, optim, inputs)
        step += 1


def out_graph_training_loop(model, optim, dataset):
    step = 0
    for inputs in dataset:
        loss = train_one_step(model, optim, inputs)
        step += 1


def main():
    model = MyModel()
    optim = keras.optimizers.Adam(1e-4)
    dataset = create_dataset()

    for i in range(5):
        t0 = time.time()
        in_graph_training_loop(model, optim, dataset)
        t1 = time.time()
        print('Time for in-graph training loop: %.3f secs' % (t1 - t0))

    print('-' * 20)

    for i in range(5):
        t0 = time.time()
        out_graph_training_loop(model, optim, dataset)
        t1 = time.time()
        print('Time for out-of-graph training loop: %.3f secs' % (t1 - t0))


if __name__ == '__main__':
    main()

@tf.function
def in_graph_training_loop(model, optim, dataset):
    step = 0
    for inputs in iter(dataset):
        loss = train_one_step(model, optim, inputs)
        step += 1

@tf.function
def in_graph_training_loop(model, optim, dataset):
    step = 0
    for inputs in iter(dataset):
        with tf.device('/device:GPU:0'):
            loss = train_one_step(model, optim, inputs)
            step += 1