from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import numpy as np
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
import tensorflow as tf
import time
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train, x_test
y_train, y_test = y_train , y_test
x_train = x_train[..., tf.newaxis].astype(np.float64)
x_test = x_test[..., tf.newaxis].astype(np.float64)
train_ds = tf.data.Dataset.from_tensor_slices(
(x_train, y_train)).shuffle(1000).batch(256)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(256)

class MyModel(Model):
    def __init__(self, *args, **kwargs):
        super(MyModel, self).__init__(args, kwargs)
        self.conv1 = Conv2D(32, 3, activation='relu')
        self.flatten = Flatten()
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(10, activation='softmax')
        
    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)
model = MyModel()
model.build((512, 28, 28, 1))
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)
@tf.function
def test_step(images, labels):
    predictions = model(images)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)

EPOCHS = 100

for epoch in range(EPOCHS):
    start = time.time()
    for images, labels in train_ds:
        train_step(tf.cast(images, tf.float32), labels)

    model.reset_metrics()
    for test_images, test_labels in test_ds:
        test_step(tf.cast(test_images, tf.float32), test_labels)
    end = time.time()
    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}, cost:{}'
    print(
        template.format(epoch + 1, train_loss.result(),
                        train_accuracy.result() * 100, test_loss.result(),
                        test_accuracy.result() * 100, end -start))

class MyModel(Model):
    def __init__(self, *args, **kwargs):
        super(MyModel, self).__init__(args, kwargs)
        self.conv1 = Conv2D(32, 3, activation='relu')
        self.flatten = Flatten()
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(10, activation='softmax')
    
    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)

class MyModel(Model):
    def __init__(self, *args, **kwargs):
        super(MyModel, self).__init__(args, kwargs)
        self.conv1 = Conv2D(32, 3, activation='relu')
        self.flatten = Flatten()
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(10, activation='softmax')

    @tf.function
    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)