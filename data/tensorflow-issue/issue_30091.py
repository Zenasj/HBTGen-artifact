from tensorflow.keras import layers

class MNISTModel(Model):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.conv1 = Conv2D(32, (3, 3), input_shape=(28, 28, 1))
        self.activ = Activation('relu')
        self.flatt = Flatten()
        self.dense = Dense(200) #Here I want to reuse this Dense() layer
        self.dense1 = Dense(200) #However, I need to re-define the same layer again to use in call()
        self.dens2 = Dense(10)

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.data import Dataset
from tensorflow.keras import datasets, Model, losses, optimizers, metrics
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Activation, MaxPooling2D


class MNISTModel(Model):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.conv1 = Conv2D(32, (3, 3), input_shape=(28, 28, 1))
        self.activ = Activation('relu')
        self.conv2 = Conv2D(32, (3, 3))
        self.maxpo = MaxPooling2D(pool_size=(2, 2))
        self.conv3 = Conv2D(64, (3, 3))
        self.flatt = Flatten()
        self.dense = Dense(200)
        self.dense1 = Dense(200)
        self.dens2 = Dense(10)

    def call(self, x):
        x = self.conv1(x)
        x = self.activ(x)
        x = self.conv2(x)
        x = self.activ(x)
        x = self.maxpo(x)

        x = self.conv3(x)
        x = self.activ(x)
        x = self.conv3(x)
        x = self.activ(x)
        x = self.maxpo(x)

        x = self.flatt(x)
        x = self.dense(x)
        x = self.activ(x)
        x = self.dense1(x)
        x = self.activ(x)
        return self.dens2(x)


(train_data, train_labels), (test_data, test_labels) = datasets.mnist.load_data()
train_data, test_data = train_data / 255.0, test_data / 255.0
train_data = train_data[..., tf.newaxis]
test_data = test_data[..., tf.newaxis]
print(train_data.shape, test_data.shape, type(train_data))
train_data = Dataset.from_tensor_slices(
    (train_data, train_labels)).shuffle(60000).batch(128)
test_data = Dataset.from_tensor_slices((test_data, test_labels)).batch(128)


model = MNISTModel()
loss_object = losses.SparseCategoricalCrossentropy()
optimizer = optimizers.Adam()

train_loss = metrics.Mean(name='train_loss')
train_accuracy = metrics.SparseCategoricalAccuracy(name='train_accuracy')
test_loss = metrics.Mean(name='test_loss')
test_accuracy = metrics.SparseCategoricalAccuracy(name='test_accuracy')


@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        logits = model(images)
        loss_value = loss_object(labels, logits)
    grads = tape.gradient(loss_value, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    train_loss(loss_value)
    train_accuracy(labels, logits)


@tf.function
def test_step(images, labels):
    logits = model(images)
    tloss_value = loss_object(labels, logits)
    test_loss(tloss_value)
    test_accuracy(labels, logits)


EPOCHS = 5

for epoch in range(EPOCHS):
    for images, labels in train_data:
        train_step(images, labels)

    for test_images, test_labels in test_data:
        test_step(test_images, test_labels)

    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    print (template.format(epoch + 1, train_loss.result(), train_accuracy.result()
                           * 100, test_loss.result(), test_accuracy.result() * 100))