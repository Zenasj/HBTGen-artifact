import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import logging

import absl.logging
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_models as tfm
from absl import flags, app
from keras import mixed_precision
from keras.utils.dataset_utils import split_dataset
from keras.utils.tf_utils import can_jit_compile

NUM_CLASSES = 104
PRNG_SEED = 42
tf.random.set_seed(PRNG_SEED)

logging.basicConfig(level=logging.DEBUG)
tf.get_logger().setLevel("DEBUG")
absl.logging.set_verbosity(absl.logging.converter.ABSL_DEBUG)

FLAGS = flags.FLAGS

flags.DEFINE_integer("batch_size", default=32, help="batch size")
flags.DEFINE_integer("epochs", default=1, help="number of epochs")


def warmup_schedule(num_training_samples):
    steps_per_epoch = int(num_training_samples / FLAGS.batch_size)
    num_train_steps = steps_per_epoch * FLAGS.epochs
    warmup_steps = int(0.4 * num_train_steps)
    initial_learning_rate = 1e-3
    
    linear_decay = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=initial_learning_rate,
        end_learning_rate=1e-4,
        decay_steps=num_train_steps,
    )
    return tfm.optimization.lr_schedule.LinearWarmup(
        warmup_learning_rate=5e-3,
        after_warmup_lr_sched=linear_decay,
        warmup_steps=warmup_steps,
    )


@tf.function(reduce_retracing=True, jit_compile=can_jit_compile())
def macro_double_soft_f1(y, y_hat):
    """Compute the macro soft F1-score as a cost (average 1 - soft-F1 across all labels).
    Use probability values instead of binary predictions.
    This version uses the computation of soft-F1 for both positive and negative class for each label.

    Args:
        y (int32 Tensor): targets array of shape (BATCH_SIZE, N_LABELS)
        y_hat (float32 Tensor): probability matrix from forward propagation of shape (BATCH_SIZE, N_LABELS)

    Returns:
        cost (scalar Tensor): value of the cost function for the batch
    """
    y = tf.cast(y, tf.bfloat16)
    y_hat = tf.cast(y_hat, tf.bfloat16)
    tp = tf.reduce_sum(y_hat * y, axis=0)
    fp = tf.reduce_sum(y_hat * (1 - y), axis=0)
    fn = tf.reduce_sum((1 - y_hat) * y, axis=0)
    tn = tf.reduce_sum((1 - y_hat) * (1 - y), axis=0)
    soft_f1_class1 = 2 * tp / (2 * tp + fn + fp + 1e-16)
    soft_f1_class0 = 2 * tn / (2 * tn + fn + fp + 1e-16)
    # reduce 1 - soft-f1_class1 in order to increase soft-f1 on class 1
    cost_class1 = 1 - soft_f1_class1
    # reduce 1 - soft-f1_class0 in order to increase soft-f1 on class 0
    cost_class0 = 1 - soft_f1_class0
    # take into account both class 1 and class 0
    cost = 0.5 * cost_class1 + cost_class0
    # average on all labels
    macro_cost = tf.reduce_mean(cost)
    return macro_cost


def main(_):
    # Unfortunately, I can't share the actual dataset due to GDPR.
    pickled_dataset = dict(
        input_word_ids=tf.constant(
            [
                [101, 15570, 15143, 49393, 15009, 40651, 0],
                [101, 18610, 23251, 107, 316053, 84805, 15954],
            ],
            dtype=tf.int32,
        ),
        input_mask=tf.constant(
            [[1, 1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1, 1]], dtype=tf.int32
        ),
        input_type_ids=tf.constant(
            [[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]], dtype=tf.int32
        ),
        # in reality, here I have multi-hot encoded labels.
        events=tf.zeros([2, NUM_CLASSES], dtype=tf.int32),
        return_events=tf.zeros([2, NUM_CLASSES], dtype=tf.int32)
    )
    
    num_training_samples = int(len(pickled_dataset["input_word_ids"]) * 0.8)
    dataset = (
        np.asarray(pickled_dataset["input_word_ids"]),
        np.asarray(pickled_dataset["input_mask"]),
        np.asarray(pickled_dataset["input_type_ids"]),
        np.asarray(pickled_dataset["events"]),
        np.asarray(pickled_dataset["return_events"]),
    )
    train_ds, val_ds = split_dataset(
        dataset, left_size=0.8, shuffle=True, seed=PRNG_SEED
    )
    
    def group_tuples(*args):
        return (
            dict(
                input_word_ids=tf.cast(args[0], tf.int32),
                input_mask=tf.cast(args[1], tf.int32),
                input_type_ids=tf.cast(args[2], tf.int32),
            ),
            (tf.cast(args[3], tf.int32), tf.cast(args[4], tf.int32)),
        )
    
    train_ds = (
        train_ds.batch(FLAGS.batch_size)
        .map(group_tuples)
        .cache()
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = (
        val_ds.batch(FLAGS.batch_size)
        .map(group_tuples)
        .cache()
        .prefetch(tf.data.AUTOTUNE)
    )
    # ----------------------------------------
    if can_jit_compile():
        tf.config.optimizer.set_jit("autoclustering")
    mixed_precision.set_global_policy("mixed_bfloat16")
    word_ids = tf.keras.layers.Input(
        shape=(None,), dtype=tf.int32, name="input_word_ids"
    )
    mask = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name="input_mask")
    type_ids = tf.keras.layers.Input(
        shape=(None,), dtype=tf.int32, name="input_type_ids"
    )
    x = hub.KerasLayer("https://tfhub.dev/google/LaBSE/2")(
        {
            "input_word_ids": word_ids,
            "input_mask": mask,
            "input_type_ids": type_ids,
        }
    )["pooled_output"]
    events = tf.keras.layers.Dense(NUM_CLASSES, activation="sigmoid", name="events")(x)
    return_events = tf.keras.layers.Dense(
        NUM_CLASSES, activation="sigmoid", name="return_events"
    )(x)
    model = tf.keras.Model(
        inputs={
            "input_word_ids": word_ids,
            "input_mask": mask,
            "input_type_ids": type_ids,
        },
        outputs=[events, return_events],
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=warmup_schedule(num_training_samples),
            jit_compile=can_jit_compile(),
        ),
        metrics={
            "events": [
                tf.keras.metrics.TruePositives(),
                tf.keras.metrics.TrueNegatives(),
                tf.keras.metrics.FalsePositives(),
                tf.keras.metrics.FalseNegatives(),
            ],
            "return_events": [
                tf.keras.metrics.TruePositives(),
                tf.keras.metrics.TrueNegatives(),
                tf.keras.metrics.FalsePositives(),
                tf.keras.metrics.FalseNegatives(),
            ],
        },
        loss={"events": macro_double_soft_f1, "return_events": macro_double_soft_f1},
        # This is the culprit.
        jit_compile=can_jit_compile(),
    )
    # ---------------------------------------
    model.fit(train_ds, validation_data=val_ds, epochs=FLAGS.epochs)
    model.save_weights("weights.keras")


if __name__ == "__main__":
    app.run(main)