from tensorflow import keras
from tensorflow.keras import layers

import numpy as np
import tensorflow as tf


class ConvLayer(tf.keras.layers.Layer):
    def call(self, image):
        shape = (3, 3, 3, 16)
        stddev = 1
        weights = tf.get_variable(
            name='weights',
            initializer=tf.truncated_normal(shape, stddev=stddev),
        )
        conv = tf.nn.conv2d(image, weights, [1, 1, 1, 1], 'SAME')
        return conv


def get_train_op(loss):
    global_step = tf.train.get_global_step()
    optimizer = tf.contrib.optimizer_v2.AdamOptimizer(.001)  # use distribution-aware optimizer
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op


def model_fn(features, labels, mode):
    image = features['image']

    # if num_layers >= 3, graph construction will error if more than 1 gpu is used
    num_layers = 3
    layer = ConvLayer()
    values = []
    for _ in xrange(num_layers):
        values.append(layer(image))
    stacked_values = tf.concat(values, axis=0)

    loss = tf.reduce_sum(stacked_values)
    mode = tf.estimator.ModeKeys.TRAIN
    train_op = get_train_op(loss)
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


def input_fn():

    shape = (100, 100, 3)

    def gen():
        while True:
            image = np.ones(shape, dtype=np.float32)
            yield ({'image': image}, [])

    ds = tf.data.Dataset.from_generator(
        gen,
        ({'image': tf.float32}, tf.float32),
        output_shapes=({'image': tf.TensorShape(shape)}, tf.TensorShape(None)),
    )
    ds = ds.repeat().batch(4)
    return ds


# top-level call
def run():
    gpus = ['/device:GPU:0', '/device:GPU:1']
    # MirroredStrategy fails
    distribution = tf.contrib.distribute.MirroredStrategy(gpus)
    # OneDeviceStrategy works
    # distribution = tf.contrib.distribute.OneDeviceStrategy(gpus[0])

    config = tf.estimator.RunConfig(
        train_distribute=distribution,
        model_dir='/path/to/output'
    )
    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=config,
    )
    estimator.train(
        input_fn=input_fn,
        steps=10,
    )

class ConvLayer(object):
    def __call__(self, image):
        with tf.variable_scope('conv_layer', reuse=tf.AUTO_REUSE):
            shape = (3, 3, 3, 16)
            stddev = 1
            weights = tf.get_variable(
                name='weights',
                initializer=tf.truncated_normal(shape, stddev=stddev),
            )
            conv = tf.nn.conv2d(image, weights, [1, 1, 1, 1], 'SAME')
            return conv