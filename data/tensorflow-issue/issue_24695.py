import random
from tensorflow import keras

import tensorflow as tf
import numpy as np
from tensorflow.keras import layers


def main(argv):
    steps = 1000
    # Generate random data
    train_x = np.random.randn(100, 4).astype('float32')
    train_y = np.random.randint(0, 3, 100).astype('int32')

    # First estimator, with *10* hidden units
    classifier = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir="tmp",
        params={'hidden': 10}
    )

    # train first model and store checkpoints on "tmp"
    classifier.train(
        input_fn=lambda: tf.data.Dataset.from_tensors((train_x, train_y)),
        steps=steps)

    # second estimator, with *20* hidden units and same model_dir output
    # BUG: should break as model_dir is the same as previous estimator!!!
    classifier = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir="tmp",
        params={'hidden': 20}
    )

    classifier.train(
        input_fn=lambda: tf.data.Dataset.from_tensors((train_x, train_y)),
        steps=steps)


def model_fn(features, labels, mode, params):

    # create model with params[hidden] units on hidden layer
    model = tf.keras.Sequential([layers.Dense(params["hidden"], activation='relu'),
                                 layers.Dense(3)])
    logits = model(features)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)