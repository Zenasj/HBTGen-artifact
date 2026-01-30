import random
from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow.compat.v1 as tf
import tensorflow as tf2
from tensorboard.plugins import projector
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Activation, Input, BatchNormalization, Layer
from tensorflow.core.protobuf import rewriter_config_pb2
import numpy as np
import tensorflow_datasets as tfds
import os

height = 5
width = 5
n_filters = 8
use_tpu = "COLAB_TPU_ADDR" in os.environ
train_batch_size = 8 * (8 if use_tpu else 1)
steps = 100
learning_rate = 1e-4
iterations_per_loop = 100
log_step_count_steps = 100
use_async_checkpointing = False
EMBEDDINGS_TENSOR_NAME = 'reshaped_head_conv_0_0'
META_DATA_FNAME = 'meta.tsv'

if use_async_checkpointing:
    save_checkpoints_steps = None
else:
    save_checkpoints_steps = max(500, iterations_per_loop)
model_dir="/content/my_storage/model"
data_dir="/content/my_storage/datasets"
gcp_project = "my_project"
tpu_zone = "us-central1"

if use_tpu:
    tpu = 'grpc://' + os.environ['COLAB_TPU_ADDR']
    tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
                                tpu, zone=tpu_zone, project=gcp_project)
    master = tpu_cluster_resolver.get_master()
else:
    tpu_cluster_resolver = None
    master = None

class Conv2d:
    def __init__(self, x, kernel, name, strides=(1, 1), padding='SAME', activation='relu', reuse=True):
        with tf.variable_scope(name, reuse=reuse):
            self.name = name
            self.x = tf.nn.conv2d(x, kernel, strides=strides, padding=padding, 
                                  name=name)
            bn_name = name + '_bn'
            self.x = tf.layers.batch_normalization(self.x,
                                              scale=False,
                                              name=bn_name)
            ac_name = name + '_ac'
            self.x = tf.nn.relu(self.x, name=ac_name)

    def get_x(self):
        return self.x

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
                    shape=(batch_size // (8 if use_tpu else 1), 32*32*n_filters,), 
                    minval=0.0, maxval=1.0, dtype=tf.dtypes.float32, seed=7777)

        def preprocess(x, y):
            x = tf.cast(x, tf.float32) * (1. / 255)
            labels_dic = {}
            for h in range(height):
                for w in range(width):
                    labels_dic["head_conv_{}_{}".format(h, w)] = y_true
            return x, labels_dic

        dataset = (x_train
                    .map(preprocess, 
                         num_parallel_calls=tf.data.experimental.AUTOTUNE)
                    .repeat()
                    .shuffle(128, seed=7777, reshuffle_each_iteration=True)
                    .batch(batch_size, drop_remainder=True)
                    .prefetch(-1))
        return dataset

    return input_fn

def get_model(features, theta, input_shape, reuse):
    with tf.variable_scope('model', reuse=reuse):
        seqs = []
        i = 0
        for h in range(height):
            seq = []
            for w in range(width):
                name = "conv_{}_{}".format(h, w)
                if seq == []:
                    if h==0 and w==0: 
                        filters = (3, 3, 3, theta[i])
                    else:
                        filters = (3, 3, theta[i-1], theta[i])
                else:
                      filters = (3, 3, theta[i-1], theta[i])
                kernel = tf.Variable(lambda: 
                                     tf.truncated_normal(filters, stddev=5e-2), 
                                     name=name)
                if seq == []:
                    if not(h==0 and w==0): 
                        features = seqs[-1][0].get_x()
                else:
                    features = seq[-1].get_x()
                seq.append(Conv2d(
                            features,
                            kernel,
                            name=name,
                            reuse=reuse))
                i += 1
            seqs.append(seq)
        outputs = []
        heads = []
        i = 0
        for seq in seqs:
            for x in seq:
                outputs.append(OutputLayer(name="output_"+x.name)(x.get_x()))
                heads.append(tf.estimator.RegressionHead(
                                               label_dimension=32*32*theta[i],
                                               name="head_"+x.name))
                i += 1
        head = tf.estimator.MultiHead(heads)
    return head, outputs

def register_embedding(embedding_tensor_name, meta_data_fname, log_dir):
  if not os.path.isdir(log_dir):
     os.makedirs(log_dir)
  config = projector.ProjectorConfig()
  embedding = config.embeddings.add()
  embedding.tensor_name = embedding_tensor_name
  embedding.metadata_path = meta_data_fname
  projector.visualize_embeddings(log_dir, config)

def save_labels_tsv(labels, filepath, log_dir):
  if not os.path.isdir(log_dir):
     os.makedirs(log_dir)
  with open(os.path.join(log_dir, filepath), 'w') as f:
    for label in labels:
      f.write('{}\n'.format(label))

def model_fn(features, labels, mode, params):
    def host_call_fn(gs, loss, lr, tensor_embeddings):
        gs = gs[0]

        with tf2.summary.create_file_writer(
            model_dir,
            max_queue=iterations_per_loop).as_default():
          with tf2.summary.record_if(True):
            tf2.summary.write(tag='projector', tensor=tensor_embeddings,
                              step=gs, name=EMBEDDINGS_TENSOR_NAME)
            tf2.summary.scalar('loss', loss[0], step=gs)
            tf2.summary.scalar('learning_rate', lr[0], step=gs)

          return tf.summary.all_v2_summary_ops()

    batch_size = params['batch_size']

    theta = width * height * [n_filters]

    assert EMBEDDINGS_TENSOR_NAME == "reshaped_head_conv_0_0"
    register_embedding("reshaped_head_conv_0_0", META_DATA_FNAME, model_dir)

    head, logits_train = get_model(features, theta, params['input_shape'], 
                                   reuse=False)
    logits_train_dic = {}
    i = 0
    for h in range(height):
        for w in range(width):
            logits_train_dic["head_conv_{}_{}".format(h, w)] = \
                tf.reshape(logits_train[i], (batch_size, 32*32*8,), 
                           name="reshaped_head_conv_{}_{}".format(h, w))
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

    gs_t = tf.reshape(tf.train.get_global_step(), [1])
    loss_t = tf.reshape(loss_op, [1])
    lr_t = tf.reshape(params['learning_rate'], [1])

    y = np.zeros((batch_size,), np.int32)
    save_labels_tsv(y, META_DATA_FNAME, model_dir)

    host_call = (host_call_fn, [gs_t, loss_t, lr_t, logits_train_dic["head_conv_0_0"]])
                
    estim_specs = tf.estimator.tpu.TPUEstimatorSpec(
                  mode=mode, loss=loss_op, train_op=train_op_fn(loss_op), 
                  host_call=host_call)
    return estim_specs

tf.logging.set_verbosity(tf.logging.INFO)
tf.disable_v2_behavior()

dataset_fn = lambda: tfds.load(name='cifar10', with_info=True, as_supervised=True, 
                               try_gcs=True, data_dir=data_dir)
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