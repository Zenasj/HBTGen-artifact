import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.estimator import model_to_estimator
import keras.backend as K
import numpy as np
import os


def model_fn():
    # 10 features
    input_layer = Input(shape=(10,), name="inputs", dtype=tf.float32)
    dense_1 = Dense(units=10, activation="relu", name="dense_1")(input_layer)
    dense_2 = Dense(units=1, activation="linear", name="outputs",
                    dtype=tf.float32)(dense_1)
    model = Model(inputs=input_layer, outputs=dense_2)
    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss="mse")
    return model


def synthetic_input_fn(num_examples, num_features):
    # dummy data
    return tf.data.Dataset.from_tensor_slices(({"inputs": np.random.random((num_examples, num_features))}, np.random.randint(10, size=num_examples))) \
        .shuffle(512) \
        .batch(32) \
        .repeat(10)


if __name__ == "__main__":

    MODEL_DIR = "./model"
    SAVED_MODEL_DIR = os.path.join(MODEL_DIR, "saved")

    def cust_train_input_fn():
        return synthetic_input_fn(1000, 10)

    def cust_eval_input_fn():
        return synthetic_input_fn(100, 10)

    tf.logging.set_verbosity(tf.logging.INFO)

    # define model and convert to estimator
    model = model_fn()
    estimator = model_to_estimator(
        keras_model=model,
        model_dir=MODEL_DIR
    )

    train_spec = tf.estimator.TrainSpec(input_fn=cust_train_input_fn)
    eval_spec = tf.estimator.EvalSpec(input_fn=cust_eval_input_fn)

    # train and evaluate dummy model
    tf.estimator.train_and_evaluate(
        estimator=estimator,
        train_spec=train_spec,
        eval_spec=eval_spec
    )

    # export all modes
    feature_spec = {"inputs": tf.placeholder(tf.float64, (None, 10))}

    label_spec = tf.placeholder(dtype=tf.int64)

    serving_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(feature_spec)
    training_fn = tf.contrib.estimator.build_raw_supervised_input_receiver_fn(feature_spec, label_spec)

    rcrv_fn_map = {
        tf.estimator.ModeKeys.TRAIN: training_fn,
        tf.estimator.ModeKeys.EVAL: training_fn,
        tf.estimator.ModeKeys.PREDICT: serving_fn
    }

    tf.contrib.estimator.export_all_saved_models(
        estimator,
        export_dir_base=SAVED_MODEL_DIR,
        input_receiver_fn_map=rcrv_fn_map
    )

    tf.keras.backend.clear_session()

    saved_estimator = tf.contrib.estimator.SavedModelEstimator(os.path.join(SAVED_MODEL_DIR, os.listdir(SAVED_MODEL_DIR)[0]))
    saved_estimator.train(cust_train_input_fn)