import random
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

model=tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(units=64, input_dim=5078, activation="relu"))
model.add(tf.keras.layers.Dense(units=32, activation="relu"))
model.add(tf.keras.layers.Dense(units=100, activation="relu"))
model.add(tf.keras.layers.Dense(units=24, activation="sigmoid"))

model.compile(optimizer="Adam", loss="binary_crossentropy", metrics=["acc"])

model.fit(X_train, y_train,
 batch_size=32,
 epochs=100, verbose=1,
 validation_split=0.15,
 shuffle=True)

def random_batch(X,y, batch_size=32):
    idx= np.random.randint(len(X), size=batch_size)
    return X[idx], y[idx]

##Further split train data to training set and validation set

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.15, random_state=1)

##Run autodiff on model

n_epochs=100
batch_size=32
n_steps=len(X_train)//batch_size

optimizer=tf.keras.optimizers.Adam()
loss=tf.keras.losses.BinaryCrossentropy()

metricLoss=tf.keras.metrics.BinaryCrossentropy()
metricsAcc=tf.keras.metrics.BinaryAccuracy()

val_acc_metric=tf.keras.metrics.BinaryAccuracy()
val_acc_loss=tf.keras.metrics.BinaryCrossentropy()


train_loss_results = []
train_accuracy_results = []

validation_loss_results = []
validation_accuracy_results = []

# for loop iterate over epochs
for epoch in range(n_epochs):

    print("Epoch {}/{}".format(epoch, n_epochs))

    # for loop iterate over batches
    for step in range(1, n_steps + 1):
        X_batch, y_batch=random_batch(X_train.values, y_train)

        # gradientTape autodiff
        with tf.GradientTape() as tape:
            y_pred=model(X_batch, training=True)
            loss_values=loss(y_batch, y_pred)
        gradients=tape.gradient(loss_values, model.trainable_weights)
        optimizer.apply_gradients(zip(gradients, model.trainable_weights))

        metricLoss(y_batch, y_pred)
        metricsAcc.update_state(y_batch, y_pred)

        # Loss and accuracy
        train_loss_results.append(loss_values)
        train_accuracy_results.append(metricsAcc.result())

        # Read out training results
        readout = 'Epoch {}, Training loss: {}, Training accuracy: {}'
        print(readout.format(epoch + 1, loss_values,
                              metricsAcc.result() * 100))

        metricsAcc.reset_states

        # Run a validation loop at the end of each epoch

    for valbatch in range(1+ n_steps +1):
        X_batchVal, y_batchVal = random_batch(X_val.values, y_val)

        val_logits = model(X_batchVal)
        # Update val metrics
        val_acc_metric(y_batchVal, val_logits)
        val_acc = val_acc_metric.result()

        val_acc_metric.update_state(y_batchVal, val_logits)

        val_loss=val_acc_loss(y_batchVal, val_logits)

        validation_loss_results.append(val_loss)
        validation_accuracy_results.append(val_acc_metric.result())

        # Read out validation results
        print( 'Validation loss: ' , float(val_loss),'Validation acc: %s' % (float(val_acc * 100),) )

        val_acc_metric.reset_states()

# -*- coding: utf8 -*-

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras as tfk


class Test(tfk.Model):
    def __init__(self):
        super(Test, self).__init__()
        self.embedding_layer = tfk.layers.Embedding(50000, 300)
        self.conv1d_layer = tfk.layers.Conv1D(256, 5)
        self.pool_layer = tfk.layers.MaxPool1D(pool_size=5, strides=2)
        self.dense1_layer = tfk.layers.Dense(128, activation=tfk.activations.relu)
        self.dense2_layer = tfk.layers.Dense(10, activation=tfk.activations.softmax)

    def call(self, inputs, training=None, mask=None):
        hidden = self.embedding_layer(inputs)
        hidden = self.conv1d_layer(hidden)
        hidden = self.pool_layer(hidden)
        hidden = tfk.layers.Flatten()(hidden)
        hidden = self.dense1_layer(hidden)
        y_pred = self.dense2_layer(hidden)
        return y_pred


class Test2(tfk.Model):
    def __init__(self):
        super(Test2, self).__init__()
        self.embedding_layer = tfk.layers.Embedding(50000, 300)
        self.rnn_layer = tfk.layers.LSTM(200)
        self.dense1_layer = tfk.layers.Dense(128, activation=tfk.activations.relu)
        self.dense2_layer = tfk.layers.Dense(10, activation=tfk.activations.softmax)

    def call(self, inputs, training=None, mask=None):
        hidden = self.embedding_layer(inputs)
        hidden = self.rnn_layer(hidden)
        hidden = self.dense1_layer(hidden)
        y_pred = self.dense2_layer(hidden)
        return y_pred


gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
epochs = 30
x = np.random.randint(low=0, high=50000, size=(10000, 128))
y = np.random.randint(low=0, high=10, size=(10000,))
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2)
trainset = tf.data.Dataset.zip(
    (tf.data.Dataset.from_tensor_slices(x_train), tf.data.Dataset.from_tensor_slices(y_train))).batch(300)
valset = tf.data.Dataset.zip(
    (tf.data.Dataset.from_tensor_slices(x_val), tf.data.Dataset.from_tensor_slices(y_val))).batch(300)
# model = Test()
model = Test2()
train_acc = tf.metrics.SparseCategoricalAccuracy()
val_acc = tf.metrics.SparseCategoricalAccuracy()
train_loss = tf.metrics.Mean()
val_loss = tf.metrics.Mean()
loss_object = tf.losses.SparseCategoricalCrossentropy()
optimizer = tf.optimizers.Adam()

model.compile(optimizer=optimizer, loss=loss_object, metrics=[train_acc])
model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=300, epochs=epochs)

# model = Test()
model = Test2()


@tf.function
def train_op(x, y):
    with tf.GradientTape() as tape:
        y_pred = model(x)
        loss = loss_object(y, y_pred)
        train_loss.update_state(loss)
        train_acc.update_state(y, y_pred)
        tf.print(y_pred)
        grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))


@tf.function
def val_op(x, y):
    y_pred = model(x)
    loss = loss_object(y, y_pred)
    val_loss.update_state(loss)
    val_acc.update_state(y, y_pred)


for epoch in range(epochs):
    print("Epoch {}/{}".format(epoch + 1, epochs))
    bar = tfk.utils.Progbar(len(y_train), unit_name="sample", stateful_metrics={"loss", "acc"})
    log_values = []
    for batch_x, batch_y in trainset:
        train_op(batch_x, batch_y)
        log_values.append(("loss", train_loss.result().numpy()))
        log_values.append(("acc", train_acc.result().numpy()))
        bar.add(len(batch_y), log_values)
    for batch_x, batch_y in valset:
        val_op(batch_x, batch_y)
    print("val_loss -", val_loss.result().numpy(), "val_acc -", val_acc.result().numpy())