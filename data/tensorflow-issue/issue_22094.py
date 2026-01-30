import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

def discriminative_loss_working(y_true, y_pred):
    # Compute the loss for only the first image in the batch
    
    prediction = y_pred[0]
    label = y_true[0]

    # Number of clusters in ground truth
    clusters,_ = tf.unique(tf.reshape(label, [-1]))

    # Compute cluster means and variances for each cluster
    def compute_mean(c):
        mask = tf.equal(label[:,:,0], c)
        masked_pixels = tf.boolean_mask(prediction, mask)
        cluster_mean = tf.reduce_mean(masked_pixels, axis=0)

        return cluster_mean

    cluster_means = tf.map_fn(compute_mean, clusters, dtype=(tf.float32))
    return tf.reduce_mean(cluster_means)

def discriminative_loss(y_true, y_pred):
    """Computes loss for a batch of images
    Args:
        y_true: (n, h, w) where each elements contains the ground truth instance id
        y_pred: (n, h, w, d) d-dimensional vector for each pixel for each image in the batch
    Returns:
        loss
    """
    # Compute the loss for each image in the batch
    def compute_loss(input):
        prediction = input[1]
        label = input[0]

        # Number of clusters in ground truth
        clusters,_ = tf.unique(tf.reshape(label, [-1]))

        # Compute cluster means and variances for each cluster
        def compute_mean(c):
            mask = tf.equal(label[:,:,0], c)
            masked_pixels = tf.boolean_mask(prediction, mask)
            cluster_mean = tf.reduce_mean(masked_pixels, axis=0)

            return cluster_mean

        cluster_means = tf.map_fn(compute_mean, clusters, dtype=(tf.float32))
        return tf.reduce_mean(cluster_means)
        
    # We want to know the loss for each image in the batch
    losses = tf.map_fn(compute_loss, (y_true,y_pred), dtype=(tf.float32))
    return losses

import tensorflow as tf
import numpy as np

def discriminative_loss(y_true, y_pred):
    """Computes loss for a batch of images
    Args:
        y_true: (n, h, w) where each elements contains the ground truth instance id
        y_pred: (n, h, w, d) d-dimensional vector for each pixel for each image in the batch
    Returns:
        loss
    """
    # Compute the loss for each image in the batch
    def compute_loss(input):
        prediction = input[1]
        label = input[0]

        # Number of clusters in ground truth
        clusters,_ = tf.unique(tf.reshape(label, [-1]))

        # Compute cluster means and variances for each cluster
        def compute_mean(c):
            mask = tf.equal(label[:,:,0], c)
            masked_pixels = tf.boolean_mask(prediction, mask)
            cluster_mean = tf.reduce_mean(masked_pixels, axis=0)

            return cluster_mean

        cluster_means = tf.map_fn(compute_mean, clusters, dtype=(tf.float32))
        return tf.reduce_mean(cluster_means)
        
    # We want to know the loss for each image in the batch
    losses = tf.map_fn(compute_loss, (y_true,y_pred), dtype=(tf.float32))
    return losses

def discriminative_loss_working(y_true, y_pred):
    # Compute the loss for only the first image in the batch
    
    prediction = y_pred[0]
    label = y_true[0]

    # Number of clusters in ground truth
    clusters,_ = tf.unique(tf.reshape(label, [-1]))

    # Compute cluster means and variances for each cluster
    def compute_mean(c):
        mask = tf.equal(label[:,:,0], c)
        masked_pixels = tf.boolean_mask(prediction, mask)
        cluster_mean = tf.reduce_mean(masked_pixels, axis=0)

        return cluster_mean

    cluster_means = tf.map_fn(compute_mean, clusters, dtype=(tf.float32))
    return tf.reduce_mean(cluster_means)

class MyModel(tf.keras.Model):
    def __init__(self, input_shape):
        super(MyModel, self).__init__()
        self.conv = tf.keras.layers.Conv2D(filters=4, kernel_size=(1,1))

    def call(self, input):
        return self.conv(input)

input_shape = (1,128,128,3)
def my_gen():
    while True:
        x = np.random.rand(1,input_shape[1], input_shape[2],3)
        y = np.random.randint(11000, 11015, (input_shape[1], input_shape[2],1))
        yield x,y

train_dataset = tf.data.Dataset.from_generator(my_gen, (tf.float32, tf.float32))
train_dataset = train_dataset.batch(1)
train_dataset = train_dataset.repeat()

model = MyModel(input_shape=input_shape)

# This is a fix to make loading weights possible
# x = tf.zeros((1,) + input_shape)
x = tf.zeros(input_shape)
y = model(x)

optimizer = tf.keras.optimizers.SGD(lr=0.0001)
model.compile(loss=discriminative_loss,optimizer=optimizer)
model.fit(train_dataset, epochs=5, steps_per_epoch=2)

import tensorflow as tf
import numpy as np

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

def discriminative_loss(y_true, y_pred):
    """Computes loss for a batch of images
    Args:
        y_true: (n, h, w) where each elements contains the ground truth instance id
        y_pred: (n, h, w, d) d-dimensional vector for each pixel for each image in the batch
    Returns:
        loss
    """
    # Compute the loss for each image in the batch
    def compute_loss(input):
        prediction = input[1]
        label = input[0]

        # Number of clusters in ground truth
        clusters,_ = tf.unique(tf.reshape(label, [-1]))

        # Compute cluster means and variances for each cluster
        def compute_mean(c):
            mask = tf.equal(label[:,:,0], c)
            masked_pixels = tf.boolean_mask(prediction, mask)
            cluster_mean = tf.reduce_mean(masked_pixels, axis=0)

            return cluster_mean

        cluster_means = tf.map_fn(compute_mean, clusters, dtype=(tf.float32))
        return tf.reduce_mean(cluster_means)

    # We want to know the loss for each image in the batch
    losses = tf.map_fn(compute_loss, (y_true,y_pred), dtype=(tf.float32))
    return losses

def discriminative_loss_working(y_true, y_pred):
    # Compute the loss for only the first image in the batch

    prediction = y_pred[0]
    label = y_true[0]

    # Number of clusters in ground truth
    clusters,_ = tf.unique(tf.reshape(label, [-1]))

    # Compute cluster means and variances for each cluster
    def compute_mean(c):
        mask = tf.equal(label[:,:,0], c)
        masked_pixels = tf.boolean_mask(prediction, mask)
        cluster_mean = tf.reduce_mean(masked_pixels, axis=0)

        return cluster_mean

    cluster_means = tf.map_fn(compute_mean, clusters, dtype=(tf.float32))
    return tf.reduce_mean(cluster_means)

class MyModel(tf.keras.Model):
    def __init__(self, input_shape):
        super(MyModel, self).__init__()
        self.conv = tf.keras.layers.Conv2D(filters=4, kernel_size=(1,1))

    def call(self, input):
        return self.conv(input)

input_shape = (1,128,128,3)
def my_gen():
    while True:
        x = np.random.rand(1,input_shape[1], input_shape[2],3)
        y = np.random.randint(11000, 11015, (input_shape[1], input_shape[2],1))
        yield x,y

train_dataset = tf.data.Dataset.from_generator(
                    my_gen,
                    (tf.float32, tf.float32),
                    (tf.TensorShape([1,128,128,3]),
                     tf.TensorShape([128,128,1])))
train_dataset = train_dataset.batch(1)
train_dataset = train_dataset.repeat()

model = MyModel(input_shape=input_shape)

# This is a fix to make loading weights possible
# x = tf.zeros((1,) + input_shape)
x = tf.zeros(input_shape)
y = model(x)

with tf.Session(config=config):
    optimizer = tf.keras.optimizers.SGD(lr=0.0001)
    model.compile(loss=discriminative_loss,optimizer=optimizer)
    model.fit(train_dataset, epochs=5, steps_per_epoch=2)