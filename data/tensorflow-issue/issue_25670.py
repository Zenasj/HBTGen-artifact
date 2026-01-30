from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
import numpy as np

from keras.datasets import mnist


# switch to example 1/2/3
EXAMPLE_CASE = 1

# flag for initial weights loading of keras model
_W_INIT = True


def dense_net(features, labels, mode, params):
    # --- code to load a keras application ---

    # commenting in this line leads to a bump in the loss everytime the
    # evaluation is run, this indicating that keras does not handle well the
    # two sessions of the estimator API
    # tf.keras.backend.set_learning_phase(mode == tf.estimator.ModeKeys.TRAIN)
    global _W_INIT

    model = tf.keras.applications.MobileNet(
        input_tensor=features,
        input_shape=(128, 128, 3),
        include_top=False,
        pooling='avg',
        weights='imagenet' if _W_INIT else None)

    # only initialize weights once
    if _W_INIT:
        _W_INIT = False

    # switch cases
    if EXAMPLE_CASE == 1:
        # model.output is the same as model.layers[-1].output
        img = model.layers[-1].output
    elif EXAMPLE_CASE == 2:
        img = model(features)
    elif EXAMPLE_CASE == 3:
        # do not use keras features
        img = tf.keras.layers.Flatten()(features)
    else:
        raise NotImplementedError

    # --- regular code from here on ---
    for units in params['dense_layers']:
        img = tf.keras.layers.Dense(units=units, activation='relu')(img)

    logits = tf.keras.layers.Dense(units=10,
                                   activation='relu')(img)

    # compute predictions
    probs = tf.nn.softmax(logits)
    predicted_classes = tf.argmax(probs, 1)

    # compute loss
    loss = tf.losses.sparse_softmax_cross_entropy(labels, logits)

    acc = tf.metrics.accuracy(labels, predicted_classes)
    metrics = {'accuracy': acc}
    tf.summary.scalar('accuarcy', acc[1])

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics)

    # create training operation
    assert mode == tf.estimator.ModeKeys.TRAIN

    optimizer = tf.train.AdagradOptimizer(learning_rate=0.01)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


def prepare_dataset(in_tuple, n):
    feats = in_tuple[0][:n, :, :]
    labels = in_tuple[1][:n]
    feats = feats.astype(np.float32)
    feats /= 255
    labels = labels.astype(np.int32)
    return (feats, labels)


def _parse_func(features, labels):
    feats = tf.expand_dims(features, -1)
    feats = tf.image.grayscale_to_rgb(feats)
    feats = tf.image.resize_images(feats, (128, 128))
    return (feats, labels)


def load_mnist(n_train=10000, n_test=3000):
    train, test = mnist.load_data()
    train = prepare_dataset(train, n_train)
    test = prepare_dataset(test, n_test)
    return train, test


def train_input_fn(imgs, labels, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((imgs, labels))
    dataset = dataset.map(_parse_func)
    dataset = dataset.shuffle(500)
    dataset = dataset.repeat().batch(batch_size)
    return dataset


def eval_input_fn(imgs, labels, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((imgs, labels))
    dataset = dataset.map(_parse_func)
    dataset = dataset.batch(batch_size)
    return dataset


def main(m_dir=None):
    # fetch data
    (x_train, y_train), (x_test, y_test) = load_mnist()

    train_spec = tf.estimator.TrainSpec(
        input_fn=lambda: train_input_fn(
            x_train, y_train, 30),
        max_steps=150)

    eval_spec = tf.estimator.EvalSpec(
        input_fn=lambda: eval_input_fn(
            x_test, y_test, 30),
        steps=100,
        start_delay_secs=0,
        throttle_secs=0)

    run_cfg = tf.estimator.RunConfig(
        model_dir=m_dir,
        tf_random_seed=2,
        save_summary_steps=2,
        save_checkpoints_steps=10,
        keep_checkpoint_max=1)

    # build network
    classifier = tf.estimator.Estimator(
        model_fn=dense_net,
        params={
            'dense_layers': [256]},
        config=run_cfg)

    # fit the model
    tf.estimator.train_and_evaluate(
        classifier,
        train_spec,
        eval_spec)


if __name__ == "__main__":
    main()

def true_positives(y_true, y_pred):
  true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
  return true_positives

false_negatives = 1.4632323
false_positives = 0.0756128
true_negatives = 126.31117
true_positives = 0.14997923