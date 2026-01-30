from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
import os
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.python.saved_model import tag_constants, signature_constants
from tensorflow.python.framework import convert_to_constants


def input_fn_builder(is_training):
    def input_fn(params):
        batch_size = 20
        num_examples = 100
        x = tf.expand_dims(tf.constant([float(i) for i in range(num_examples)], dtype=tf.float32), axis=1)
        y = 2 * x + 1
        d = tf.data.Dataset.from_tensor_slices({"x": x, "y": y})

        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=5)

        d = d.batch(batch_size=batch_size)
        return d

    return input_fn


def model_fn_builder(optimizer):
    def model_fn(features, labels, mode, params):
        x = features["x"]
        if "y" in features:
            y = features["y"]
        else:
            y = labels
        model = tf.keras.Sequential([tf.keras.layers.Dense(1)])

        def loss_fn():  # For training stage only.
            out = model(x, training=True)
            loss = tf.keras.losses.mean_squared_error(y_true=y, y_pred=out)
            loss = tf.reduce_mean(loss)
            return loss

        training = (mode == tf.estimator.ModeKeys.TRAIN)
        model.trainable = training
        out = model(x, training=False)
        if mode == tf.estimator.ModeKeys.PREDICT:
            pred_dict = {"out": out}
            return tf.estimator.EstimatorSpec(mode, pred_dict)
        loss = tf.keras.losses.mean_squared_error(y_true=y, y_pred=out)
        loss = tf.reduce_mean(loss)
        train_op = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer.iterations = tf.compat.v1.train.get_or_create_global_step()
            trainable_variables = model.trainable_variables
            train_op = optimizer.minimize(loss_fn, var_list=trainable_variables)
        return tf.estimator.EstimatorSpec(mode, predictions=out, loss=loss, train_op=train_op,
                                          eval_metric_ops={})

    return model_fn


output_dir = "./trt_test"
checkpoint_dir = os.path.join(output_dir, "checkpoint")
saved_model_dir = os.path.join(output_dir, "saved_model")
trt_saved_model_dir = os.path.join(output_dir, "trt_saved_model")

# Train a simple regression model
optimizer = tf.optimizers.SGD(learning_rate=5e-7)
run_config = tf.estimator.RunConfig(model_dir=output_dir, save_checkpoints_steps=100, keep_checkpoint_max=5)

estimator = tf.estimator.Estimator(model_fn_builder(optimizer),
                                   config=run_config, params={})
train_spec = tf.estimator.TrainSpec(input_fn_builder(is_training=True), max_steps=1000)
eval_spec = tf.estimator.EvalSpec(input_fn_builder(is_training=False), throttle_secs=0)
tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
print("Training complete")
# Export as saved_model
features = {"x": tf.keras.Input(shape=(1,), dtype=tf.float32)}
serving_input_receiver_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(features=features)
estimator.export_saved_model(saved_model_dir, serving_input_receiver_fn)
# Convert to TensorRT

input_saved_model_dir = os.path.join(saved_model_dir, os.listdir(saved_model_dir)[0])

converter = trt.TrtGraphConverterV2(input_saved_model_dir=input_saved_model_dir)
converter.convert()
converter.save(trt_saved_model_dir)

saved_model_loaded = tf.saved_model.load(
    trt_saved_model_dir, tags=[tag_constants.SERVING])
print(saved_model_loaded.signatures)

graph_func = saved_model_loaded.signatures[
    signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
frozen_func = convert_to_constants.convert_variables_to_constants_v2(
    graph_func)


def wrap_func(*args, **kwargs):
    # Assumes frozen_func has one output tensor
    return frozen_func(*args, **kwargs)[0]


x = tf.constant([[1.]])
# output = wrap_func(x)  # Error, The function demands "global_step" besides parameter "input_1"
output = wrap_func(tf.constant(0), x)  # Error, cannot feed a resource tensor with scalar

def build_serving_input_receiver_fn(default_batch_size=None):
    def serving_input_receiver_fn():
        """A serving_input_receiver_fn that expects features to be fed directly."""
        receiver_tensors = {"x": tf.compat.v1.placeholder(shape=(default_batch_size, 1,), dtype=tf.float32)}
        return tf.estimator.export.ServingInputReceiver(receiver_tensors, receiver_tensors)
    return serving_input_receiver_fn


serving_input_receiver_fn = build_serving_input_receiver_fn(default_batch_size=None)

import tensorflow as tf
import os
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.python.saved_model import tag_constants, signature_constants
from tensorflow.python.framework import convert_to_constants


def input_fn_builder(is_training):
    def input_fn(params):
        batch_size = 20
        num_examples = 100
        x = tf.expand_dims(tf.constant([float(i) for i in range(num_examples)], dtype=tf.float32), axis=1)
        y = 2 * x + 1
        d = tf.data.Dataset.from_tensor_slices({"x": x, "y": y})

        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=5)

        d = d.batch(batch_size=batch_size)
        return d

    return input_fn


def model_fn_builder(optimizer):
    def model_fn(features, labels, mode, params):
        x = features["x"]
        if "y" in features:
            y = features["y"]
        else:
            y = labels
        model = tf.keras.Sequential([tf.keras.layers.Dense(1)])

        def loss_fn():  # For training stage only.
            out = model(x, training=True)
            loss = tf.keras.losses.mean_squared_error(y_true=y, y_pred=out)
            loss = tf.reduce_mean(loss)
            return loss

        training = (mode == tf.estimator.ModeKeys.TRAIN)
        model.trainable = training
        out = model(x, training=False)
        if mode == tf.estimator.ModeKeys.PREDICT:
            pred_dict = {"out": out}
            return tf.estimator.EstimatorSpec(mode, pred_dict)
        loss = tf.keras.losses.mean_squared_error(y_true=y, y_pred=out)
        loss = tf.reduce_mean(loss)
        train_op = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer.iterations = tf.compat.v1.train.get_or_create_global_step()
            trainable_variables = model.trainable_variables
            train_op = optimizer.minimize(loss_fn, var_list=trainable_variables)
        return tf.estimator.EstimatorSpec(mode, predictions=out, loss=loss, train_op=train_op,
                                          eval_metric_ops={})

    return model_fn


output_dir = "./trt_test"
checkpoint_dir = os.path.join(output_dir, "checkpoint")
saved_model_dir = os.path.join(output_dir, "saved_model")
trt_saved_model_dir = os.path.join(output_dir, "trt_saved_model")

# Train a simple regression model
optimizer = tf.optimizers.SGD(learning_rate=5e-7)
run_config = tf.estimator.RunConfig(model_dir=output_dir, save_checkpoints_steps=100, keep_checkpoint_max=5)

estimator = tf.estimator.Estimator(model_fn_builder(optimizer),
                                   config=run_config, params={})
train_spec = tf.estimator.TrainSpec(input_fn_builder(is_training=True), max_steps=1000)
eval_spec = tf.estimator.EvalSpec(input_fn_builder(is_training=False), throttle_secs=0)
tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
print("Training complete")
# Export as saved_model
#features = {"x": tf.keras.Input(shape=(1,), dtype=tf.float32)}
#serving_input_receiver_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(features=features)


def build_serving_input_receiver_fn(default_batch_size=None):
    def serving_input_receiver_fn():
        """A serving_input_receiver_fn that expects features to be fed directly."""
        receiver_tensors = {"x": tf.compat.v1.placeholder(shape=(default_batch_size, 1,), dtype=tf.float32)}
        return tf.estimator.export.ServingInputReceiver(receiver_tensors, receiver_tensors)
    return serving_input_receiver_fn


serving_input_receiver_fn = build_serving_input_receiver_fn()


estimator.export_saved_model(saved_model_dir, serving_input_receiver_fn)
# Convert to TensorRT

input_saved_model_dir = os.path.join(saved_model_dir, list(filter(lambda x: not x.startswith("temp"), os.listdir(saved_model_dir)))[0])

converter = trt.TrtGraphConverterV2(input_saved_model_dir=input_saved_model_dir)
converter.convert()
converter.save(trt_saved_model_dir)

saved_model_loaded = tf.saved_model.load(
    trt_saved_model_dir, tags=[tag_constants.SERVING])
print(saved_model_loaded.signatures)

graph_func = saved_model_loaded.signatures[
    signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
frozen_func = convert_to_constants.convert_variables_to_constants_v2(
    graph_func)


def wrap_func(*args, **kwargs):
    # Assumes frozen_func has one output tensor
    return frozen_func(*args, **kwargs)[0]

x = tf.constant([[1.]])
output = wrap_func(x)  # Error, The function demands "global_step" besides parameter "input_1"
# output = wrap_func(tf.constant(0), x)  # Error, cannot feed a resource tensor with scalar