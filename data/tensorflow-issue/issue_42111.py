import random
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import logging
import os
import shutil
import sys

import numpy as np
import tensorflow as tf

def setup_multi_node_training():
    # IMPORTANT: SET UP TF_CONFIG FOR MULTINODE TRAINING HERE
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    tf.config.set_soft_device_placement(True)
    mirrored_strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy(tf.distribute.experimental.CollectiveCommunication.NCCL)
    # Constructs the configuration
    run_config = tf.estimator.RunConfig(
        train_distribute=mirrored_strategy,
    )
    return run_config

def input_fn():
    dataset = tf.data.Dataset.from_tensors([tf.random.normal(shape=[496, 496, 64])] * 3)
    dataset = dataset.repeat()
    return dataset

def batch_norm(x, is_training):
    layer = tf.keras.layers.experimental.SyncBatchNormalization(axis=-1)
    x_norm = layer(x, is_training)
    with tf.control_dependencies(layer.get_updates_for(x)):
        x_norm = tf.identity(x_norm)
    return x_norm

def inference(features, is_training):
    conv1 = tf.keras.layers.Conv2D(32, 3, padding="SAME")(features)
    conv1bn = batch_norm(conv1, is_training)
    deconv1bn = batch_norm(conv1bn, is_training)
    conv2 = tf.keras.layers.Conv2D(32, 3, padding="SAME")(conv1bn)
    conv2bn = batch_norm(conv2, is_training)
    return tf.keras.layers.Concatenate()([conv1bn, deconv1bn, conv2bn])

def compute_loss(predictions, labels, is_training):
    return tf.reduce_mean(predictions)

def model_fn(features, labels, mode):
    global_step = tf.compat.v1.train.get_global_step()
    is_training = mode == tf.estimator.ModeKeys.TRAIN

    predictions = inference(features, is_training)
    loss = compute_loss(predictions, labels, is_training)

    optimizer = tf.compat.v1.train.GradientDescentOptimizer(1e-5)
    train_op = optimizer.minimize(loss, global_step=global_step)

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

def main():
    model_dir = "/tmp/output"
    run_config_params = {
        "save_checkpoints_steps": 100,
        "save_summary_steps": 100,
        "log_step_count_steps": 100,
        "tf_random_seed": 0,
        "keep_checkpoint_max": 1,
        "model_dir": model_dir,
    }
    run_config = setup_multi_node_training().replace(**run_config_params)
    estimator = tf.estimator.Estimator(model_fn=model_fn, config=run_config)

    train_spec = tf.estimator.TrainSpec(input_fn=input_fn, max_steps=1000)

    eval_spec = tf.estimator.EvalSpec(
        input_fn=input_fn, steps=100, throttle_secs=0, start_delay_secs=0
    )

    print("Training and evaluating model...")
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

if __name__ == "__main__":
    main()

import os

import tensorflow as tf
from tensorflow import keras

class MyModel(keras.Model):
    def __init__(self, **kwargs):
        super().__init__(name="my_model", **kwargs)
        self.conv1 = keras.layers.Conv2D(16, 3, padding="SAME")
        self.sbn1 = keras.layers.experimental.SyncBatchNormalization()
        self.sbn2 = keras.layers.experimental.SyncBatchNormalization()
        self.conv2 = keras.layers.Conv2D(32, 3, padding="SAME")
        self.sbn3 = keras.layers.experimental.SyncBatchNormalization()
        self.concat = keras.layers.Concatenate()

    def call(self, inputs, training=False):
        conv1 = self.conv1(inputs)
        conv1bn = self.sbn1(conv1, training)
        conv1bn2 = self.sbn2(conv1bn, training)
        conv2 = self.conv2(conv1bn)
        conv2bn = self.sbn3(conv2, training)
        return self.concat([conv1bn, conv1bn2, conv2bn])

def get_dataset():
    dataset = tf.data.Dataset.from_tensors(
        [tf.random.normal(shape=[496, 496, 64])] * 3
    )
    dataset = dataset.repeat()
    dataset = tf.data.Dataset.zip((dataset, dataset))
    return dataset

def main():
    model_dir = "/tmp/keras_example"

    # IMPORTANT: SET UP TF_CONFIG FOR MULTINODE TRAINING HERE
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    tf.config.set_soft_device_placement(True)
    mirrored_strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy(tf.distribute.experimental.CollectiveCommunication.NCCL)

    # Create dataset
    train_dataset = get_dataset()

    with strategy.scope():
        model = MyModel()

        model.compile(
            optimizer=keras.optimizers.Adam(),
            loss=keras.losses.MeanSquaredError(),
        )

    model.fit(x=train_dataset, steps_per_epoch=100, epochs=1)

if __name__ == "__main__":
    main()

def inference(features, is_training):
    conv1 = tf.keras.layers.Conv2D(32, 3, padding="SAME")(features)
    conv1bn = batch_norm(conv1, is_training)
    with tf.control_dependencies([conv1bn]):
        deconv1bn = batch_norm(conv1bn, is_training)
    conv2 = tf.keras.layers.Conv2D(32, 3, padding="SAME")(conv1bn)
    with tf.control_dependencies([deconv1bn]):
        conv2bn = batch_norm(conv2, is_training)
    return tf.keras.layers.Concatenate()([conv1bn, deconv1bn, conv2bn]) # must execute in this order

def dependency(x, y):
    """Returns an instance of x that nominally depends on y (but in reality
    is identical to x).
    """
    return x + 0 * tf.reduce_mean(y)

def inference(features, is_training):
    conv1 = tf.keras.layers.Conv2D(32, 3, padding="SAME")(features)
    conv1bn = batch_norm(conv1, is_training)
    deconv1bn = batch_norm(conv1bn, is_training)
    # Create fake dependency on deconv1bn to enforce ordering
    conv1bn = dependency(conv1bn, deconv1bn)
    conv2 = tf.keras.layers.Conv2D(32, 3, padding="SAME")(conv1bn)
    conv2bn = batch_norm(conv2, is_training)
    return tf.keras.layers.Concatenate()([conv1bn, deconv1bn, conv2bn])

def inference(features, is_training):
    conv1 = tf.keras.layers.Conv2D(32, 3, padding="SAME")(features)
    conv1bn = batch_norm(conv1, is_training)
    deconv1bn = batch_norm(conv1bn, is_training)
    with tf.control_dependencies([deconv1bn]):
        conv2 = tf.keras.layers.Conv2D(32, 3, padding="SAME")(conv1bn)
    conv2bn = batch_norm(conv2, is_training)
    return tf.keras.layers.Concatenate()([conv1bn, deconv1bn, conv2bn])