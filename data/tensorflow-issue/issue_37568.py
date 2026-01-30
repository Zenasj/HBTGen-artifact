import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

import tensorflow.compat.v1 as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Activation, Input, BatchNormalization, Layer
from tensorflow.core.protobuf import rewriter_config_pb2
import numpy as np
import tensorflow_datasets as tfds

height = 5
width = 5
use_tpu = True
train_batch_size = 8 * 8 if use_tpu else 1
steps = 20000
learning_rate = 1e-4
iterations_per_loop = 100
log_step_count_steps = 100
use_async_checkpointing = False
if use_async_checkpointing:
    save_checkpoints_steps = None
else:
    save_checkpoints_steps = max(500, iterations_per_loop)
model_dir="gs://my_storage/model"
data_dir="gs://my_storage/datasets"
tpu = "grpc://10.3.101.2:8470"
gcp_project = "my_project"
tpu_zone = "us-central1"

if use_tpu:
    tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
                                tpu, zone=tpu_zone, project=gcp_project)
    master = tpu_cluster_resolver.get_master()
else:
    tpu_cluster_resolver = None
    master = None

class Conv2d:
    def __init__(self, x, filters, kernel_size, name, strides=(1, 1), padding='same', activation='relu', reuse=True):
        with tf.variable_scope(name, reuse=reuse):
            self.name = name
            self.x = Conv2D(filters, kernel_size, strides=strides, padding=padding, name=name)(x)
            bn_name = name + '_bn'
            self.x = BatchNormalization(scale=False,
                                        name=bn_name)(self.x)
            ac_name = name + '_ac'
            self.x = Activation(activation=activation, name=ac_name)(self.x)

class OutputLayer(Layer):
    def __init__(self, name, **kwargs):
        super(OutputLayer, self).__init__(name=name, **kwargs)

    def call(self, inputs):
        return inputs

def make_input_fn(dataset_fn, params):

    def input_fn(params):
        x_train = dataset_fn()[0]["train"]
        batch_size = params["batch_size"]
        y_true = tf.random.uniform(
                    shape=(8*batch_size, 32*32*8,), minval=0.0, maxval=1.0, dtype=tf.dtypes.float32, seed=7777)

        def preprocess(x, y):
            x = tf.cast(x, tf.float32) * (1. / 255)
            labels_dic = {}
            for h in range(height):
                for w in range(width):
                    labels_dic["head_conv_{}_{}".format(h, w)] = y_true
            return x, labels_dic

        dataset = (x_train
                    .map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
                    .repeat()
                    .shuffle(128, seed=7777, reshuffle_each_iteration=True)
                    .batch(8*batch_size, drop_remainder=True)
                    .prefetch(-1))
        return dataset

    return input_fn

def get_model(features, input_shape, reuse):
    with tf.variable_scope('model', reuse=reuse):
        inputs = Input(shape=input_shape)
        seqs = []
        n_filters = 8
        for h in range(height):
            seq = []
            for w in range(width):
                if seq == []:
                   if h==0 and w==0: 
                        seq.append(Conv2d(inputs, n_filters, (3, 3), name="conv_{}_{}".format(h, w), reuse=reuse))
                   else:
                        seq.append(Conv2d(seqs[-1][0].x, n_filters, (3, 3), name="conv_{}_{}".format(h, w), reuse=reuse))
                else:
                    seq.append(Conv2d(seq[-1].x, n_filters, (3, 3), name="conv_{}_{}".format(h, w), reuse=reuse))
            seqs.append(seq)
        tmp = np.array([x for x in [seq for seq in seqs]]).ravel()
        outputs = []
        heads = []
        for x in tmp:
            outputs.append(OutputLayer(name="output_"+x.name)(x.x))
            heads.append(tf.estimator.RegressionHead(label_dimension=32*32*8, name="head_"+x.name))

        model = Model(inputs=inputs, outputs=outputs)
        head = tf.estimator.MultiHead(heads)
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        metrics = ['accuracy']
        model.compile(loss='mean_squared_error',
                          optimizer=opt,
                          metrics=metrics)
        model.summary()
    return model, head

def model_fn(features, labels, mode, params):
    batch_size = 8 * params['batch_size']

    model, head = get_model(features, params['input_shape'], reuse=False)
    logits_train = model(features)
    logits_train_dic = {}
    i = 0
    for h in range(height):
        for w in range(width):
            logits_train[i] = tf.reshape(logits_train[i], (batch_size, 32*32*8,))
            logits_train_dic["head_conv_{}_{}".format(h, w)] = logits_train[i]
            i += 1
    pred_classes = tf.argmax(logits_train, axis=1)
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.tpu.TPUEstimatorSpec(mode, predictions=pred_classes)

    new_labels = {}
    for key in labels:
        new_labels[key] = labels[key][0]
    loss = 0.0
    for key in logits_train_dic:
        logit_train = logits_train_dic[key]
        loss += tf.square(labels[key]-logit_train)
    loss_op = tf.reduce_mean(loss)
    optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate'])
    if params['use_tpu']:
        optimizer = tf.tpu.CrossShardOptimizer(optimizer)
    train_op_fn = lambda loss_op: optimizer.minimize(
                                  loss_op,
                                  global_step=tf.train.get_global_step())
                
    estim_specs = head.create_estimator_spec(
                  features={"x": features},
                  labels=new_labels,
                  mode=mode,
                  logits=logits_train_dic,
                  train_op_fn=train_op_fn)
    return estim_specs

tf.logging.set_verbosity(tf.logging.INFO)
tf.disable_v2_behavior()

dataset_fn = lambda: tfds.load(
            name='cifar10',
            with_info=True,
            as_supervised=True,
            try_gcs=True,
            data_dir=data_dir)
info = dataset_fn()[1]
n_samples = info.splits['train'].get_proto().statistics.num_examples
n_classes = info.features['label'].num_classes
train_shape = info.features['image'].shape
tf.config.set_soft_device_placement(True)

config = tf.estimator.tpu.RunConfig(
              master=master,
              model_dir=model_dir,
              save_checkpoints_steps=save_checkpoints_steps,
              log_step_count_steps=log_step_count_steps,
              session_config=tf.ConfigProto(
                  graph_options=tf.GraphOptions(
                      rewrite_options=rewriter_config_pb2.RewriterConfig(
                          disable_meta_optimizer=True))),
              tpu_config=tf.estimator.tpu.TPUConfig(
                  iterations_per_loop=iterations_per_loop,
                  per_host_input_for_training=tf.estimator.tpu.InputPipelineConfig
                  .PER_HOST_V2))

params = {
    'use_tpu': use_tpu,
    'input_shape': train_shape,
    'learning_rate': learning_rate
}

model = tf.estimator.tpu.TPUEstimator(
          model_fn, use_tpu=use_tpu,
          config=config,
          train_batch_size=train_batch_size,
          params=params)
model.train(make_input_fn(dataset_fn, params), steps=steps)

estim_specs = head.create_estimator_spec(
                  features={"x": features},
                  labels=new_labels,
                  mode=mode,
                  logits=logits_train_dic,
                  train_op_fn=train_op_fn)

estim_specs = tf.estimator.tpu.TPUEstimatorSpec(
                  mode=mode, loss=loss_op, train_op=train_op_fn(loss_op))