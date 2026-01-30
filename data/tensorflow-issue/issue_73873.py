import math
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

import time
import sys
import numpy as np
import itertools
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential, Model
from random import shuffle, random

tf.random.set_seed(1234)
np.random.seed(1234)

np.set_printoptions(threshold=sys.maxsize)

mnist = tf.keras.datasets.mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

img_A_input = Input((28, 28, 3), name='img_A_input')
img_B_input = Input((28, 28, 3), name='img_B_input')


def euclidean_distance(vects):
    """Find the Euclidean distance between two vectors.

    Arguments:
        vects: List containing two tensors of same length.

    Returns:
        Tensor containing euclidean distance
        (as floating point value) between vectors.
    """

    x, y = vects
    sum_square = tf.math.reduce_sum(tf.math.square(x - y), axis=1, keepdims=True)
    return tf.math.sqrt(tf.math.maximum(sum_square, tf.keras.backend.epsilon()))

def loss(margin=1):
    """Provides 'contrastive_loss' an enclosing scope with variable 'margin'.

    Arguments:
        margin: Integer, defines the baseline for distance for which pairs
                should be classified as dissimilar. - (default is 1).

    Returns:
        'contrastive_loss' function with data ('margin') attached.
    """

    # Contrastive loss = mean( (1-true_value) * square(prediction) +
    #                         true_value * square( max(margin-prediction, 0) ))
    def contrastive_loss(y_true, y_pred):
        """Calculates the contrastive loss.

        Arguments:
            y_true: List of labels, each label is of type float32.
            y_pred: List of predictions of same length as of y_true,
                    each label is of type float32.

        Returns:
            A tensor containing contrastive loss as floating point value.
        """

        square_pred = tf.math.square(y_pred)
        margin_square = tf.math.square(tf.math.maximum(margin - (y_pred), 0))
        return tf.math.reduce_mean((1 - y_true) * square_pred + (y_true) * margin_square)

    return contrastive_loss

cnn = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (2, 2), activation="tanh", padding="same"),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Conv2D(64, (2, 2), activation="tanh", padding="same"),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Conv2D(128, (2, 2), activation="tanh", padding="same"),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Conv2D(256, (2, 2), activation="tanh", padding="same"),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(64, activation='tanh')
])

feature_vector_A = cnn(img_A_input)
feature_vector_B = cnn(img_B_input)

merge_layer = tf.keras.layers.Lambda(euclidean_distance, output_shape=(1,))(
    [feature_vector_A, feature_vector_B]
)
normal_layer = tf.keras.layers.BatchNormalization()(merge_layer)

output = Dense(1, activation='sigmoid')(normal_layer)

model = Model(inputs=[img_A_input, img_B_input], outputs=output)

random_indices = np.random.choice(X_train.shape[0], 500, replace=False)
X_train_sample, y_train_sample = X_train[random_indices], y_train[random_indices]

random_indices = np.random.choice(X_test.shape[0], 200, replace=False)
X_test_sample, y_test_sample = X_test[random_indices], y_test[random_indices]


def make_paired_dataset(X,y):
    X_pairs, y_pairs = [], []

    tuples = [(x1, y1) for x1, y1 in zip(X,y)]

    for t in itertools.product(tuples, tuples):
        img_A, label_A = t[0]
        img_B, label_B = t[1]

        img_A = tf.expand_dims(img_A, -1)
        img_A = tf.image.grayscale_to_rgb(img_A)

        img_B = tf.expand_dims(img_B, -1)
        img_B = tf.image.grayscale_to_rgb(img_B)

        new_label = float(label_A == label_B)

        X_pairs.append([img_A, img_B])
        y_pairs.append(new_label)

    pairs = [(x, y) for x, y in zip(X_pairs, y_pairs)]
    shuffle(pairs)

    X_pairs = np.array([x for x, _ in pairs])
    y_pairs = np.array([y for _, y in pairs])

    return X_pairs, y_pairs

def generate_paired_samples_dev(X, y):
    tuples = [(x1, y1) for x1, y1 in zip(X, y)]

    for t in itertools.product(tuples, tuples):
        img_A, label_A = t[0]
        img_B, label_B = t[1]

        img_A = tf.expand_dims(img_A, -1)
        img_A = tf.image.grayscale_to_rgb(img_A)

        img_B = tf.expand_dims(img_B, -1)
        img_B = tf.image.grayscale_to_rgb(img_B)

        new_label = float(label_A == label_B)
        yield [img_A, img_B], new_label


X_train_pairs, y_train_pairs = make_paired_dataset(X_train_sample, y_train_sample)
X_test_pairs, y_test_pairs = make_paired_dataset(X_test_sample, y_test_sample)

train_dataset = tf.data.Dataset.from_generator(
    generate_paired_samples_dev,
    args=(X_train_sample, y_train_sample),
    output_signature=(
        tf.TensorSpec(shape=(2,) + (28, 28, 3), dtype=tf.int32),
        tf.TensorSpec(shape=(), dtype=tf.float32)
    )
).map(lambda x, y: ({'img_A_input': x[0], 'img_B_input': x[1]}, y))
train_dataset = train_dataset.batch(batch_size=32)
train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_generator(
    generate_paired_samples_dev,
    args=(X_test_sample, y_test_sample),
    output_signature=(
        tf.TensorSpec(shape=(2,) + (28, 28, 3), dtype=tf.int32),
        tf.TensorSpec(shape=(), dtype=tf.float32)
    )
).map(lambda x, y: ({'img_A_input': x[0], 'img_B_input': x[1]}, y))
val_dataset = val_dataset.batch(batch_size=32)
val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)


model.compile(loss=loss(), optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])
model.summary()

class_weight = {0: 0.1,
                1: 0.9}

weights = model.get_weights()

"""
    Training with all the data pairs stored in arrays
"""
model.fit(x=[X_train_pairs[:, 0, :, :], X_train_pairs[:, 1, :, :]], # numbers to differentiate the input images for each subnetwork
          y=y_train_pairs,
          validation_data=([X_test_pairs[:, 0, :, :], X_test_pairs[:, 1, :, :]], y_test_pairs),
          epochs=10, batch_size=32, class_weight=class_weight, verbose=2)

print(model.evaluate(x=[X_test_pairs[:, 0, :, :], X_test_pairs[:, 1, :, :]], y=y_test_pairs, batch_size=32, verbose=2))

model.set_weights(weights) # just to reset the initial state without any training

"""
    Training with all the data using the tf.data.Dataset class, so that the data isn't all kept in memory at the same time
"""
model.fit(train_dataset,
          validation_data=val_dataset,
          epochs=10, class_weight=class_weight, verbose=2)
print(model.evaluate(x=[X_test_pairs[:, 0, :, :], X_test_pairs[:, 1, :, :]], y=y_test_pairs, batch_size=32, verbose=2))

model.set_weights(weights) # just to reset the initial state without any training


batch_size = 32
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
loss_fn = loss()

train_acc_metric = tf.keras.metrics.Accuracy()
val_acc_metric = tf.keras.metrics.Accuracy()

"""
    Custom training loop to train the model by batch. This is just a demo where the final use case is to only have in memory
    the file paths for the images, and load only each batch of images when needed, since the whole dataset wouldn't fit in memory
"""
for epoch in range(10):
    tmp = [(x, y) for x, y in zip(X_train_pairs, y_train_pairs)]
    shuffle(tmp)
    X_train_pairs = np.array([x for x, _ in tmp])
    y_train_pairs = np.array([y for _, y in tmp])

    print("Starting epoch " + str(epoch + 1))
    start_time = time.time()
    for idx in range((len(y_train_pairs) // batch_size)):
        batch_x = X_train_pairs[idx * batch_size: (idx + 1) * batch_size]
        batch_y = y_train_pairs[idx * batch_size: (idx + 1) * batch_size]
        with tf.GradientTape() as tape:
            preds = model([batch_x[:, 0, :, :], batch_x[:, 1, :, :]], training=True)
            loss = loss_fn(batch_y, preds)
        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        train_acc_metric.update_state(batch_y, preds)

        # Log every 200 batches.
        if idx % 200 == 0:
            print(
                "Training loss (for one batch) at step %d: %.4f"
                % (idx + 1, float(loss))
            )
            print("Seen so far: %s samples" % ((idx + 1) * batch_size))

    # Display metrics at the end of each epoch.
    train_acc = train_acc_metric.result()
    print("Training acc over epoch: %.4f" % (float(train_acc),))

    # Reset training metrics at the end of each epoch
    train_acc_metric.reset_states()

    # Run a validation loop at the end of each epoch.
    for idx in range((len(y_test_pairs) // batch_size)):
        batch_x = X_test_pairs[idx * batch_size: (idx + 1) * batch_size]
        batch_y = y_test_pairs[idx * batch_size: (idx + 1) * batch_size]
        val_preds = model([batch_x[:, 0, :, :], batch_x[:, 1, :, :]], training=False)
        val_acc_metric.update_state(batch_y, val_preds)

    val_acc = val_acc_metric.result()
    val_acc_metric.reset_states()
    print("Validation acc: %.4f" % (float(val_acc),))
    print("Time taken: %.2fs" % (time.time() - start_time))