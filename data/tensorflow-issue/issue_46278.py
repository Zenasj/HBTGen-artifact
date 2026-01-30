import random
from tensorflow.keras import optimizers

import datetime
import math
import os
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing


def parse_record_batch(message, features):
  parsed_feature_dict = tf.io.parse_example(message, features)
  label = parsed_feature_dict.pop('label')
  weight = parsed_feature_dict.pop('weight')
  # We have to padding all sparse feature with -1 in sync with recommend engine which also padding -1 before sending grpc predict request to TF serving.
  for feature_name, cur_tensor in parsed_feature_dict.items():
    if feature_name.startswith('sparse'):
      parsed_feature_dict[feature_name] = tf.sparse.to_dense(cur_tensor, -1)

  return parsed_feature_dict, label, weight

def input_fn(file_path, batch_size, num_epochs, features):
  # We need to set shuffle to False because multi worker stragtegy will auto split training file, so no shuffle means split across worker is in the same certain same way
  dataset = tf.data.Dataset.list_files(file_path, shuffle=False)
  dataset = dataset.interleave(
    lambda filename: tf.data.TFRecordDataset(filename),
    cycle_length=10,
    block_length=1000,
    num_parallel_calls=10
  )
  dataset = dataset.repeat(num_epochs).shuffle(batch_size * 20, reshuffle_each_iteration=True)
  dataset = dataset.batch(batch_size=batch_size)
  dataset = dataset.map(lambda x: parse_record_batch(x, features), num_parallel_calls=5)
  return dataset

def get_nhoursago(n, work_hour=None):
  if work_hour:
    date = time.strptime(work_hour, '%Y-%m-%d-%H')
    day = datetime.datetime(
      date[0], date[1], date[2], date[3]) - datetime.timedelta(hours=n)
  else:
    day = datetime.datetime.now() - datetime.timedelta(hours=n)
  return day.strftime('%Y-%m-%d-%H')

def get_nhourslast(n, work_hour=None):
  if work_hour:
    date = time.strptime(work_hour, '%Y-%m-%d-%H')
    day = datetime.datetime(
      date[0], date[1], date[2], date[3]) + datetime.timedelta(hours=n)
  else:
    day = datetime.datetime.now() + datetime.timedelta(hours=n)
  return day.strftime('%Y-%m-%d-%H')

# get tfrecords from HDFS, HDFS path format is:
# father_directory/2020-01-09-01/tfrecords/part-*
#                            .../data_size        (file which record the tfrecords sample length)
def get_train_file_info(data_dir, biz_date, data_range):
  date_list = [d.strftime("%Y-%m-%d-%H") for d in
               pd.date_range(get_nhoursago(data_range, biz_date), get_nhoursago(1, biz_date),
                             freq='H')]
  data_path = []
  sample_count = 0
  for cur_data in data_dir.split(','):
    data_path.extend(['%s/%s/tfrecords/part-*' % (cur_data, dt) for dt in date_list])
    sample_count = sample_count + sum([int(tf.io.gfile.GFile('%s/%s/data_size' % (cur_data, dt), "r").readline()) for dt in date_list])
  return data_path, sample_count


def get_cur_time():
  return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def get_parse_example_config(all_dict, feature_name):
  if feature_name.startswith('sparse'):
    all_dict[feature_name] = tf.io.VarLenFeature(tf.int64)
  else:
    all_dict[feature_name] = tf.io.FixedLenFeature((1), tf.float32, 0.0)

class DenseToRaggedLayer(tf.keras.layers.Layer):
  def __init__(self, ignore_value=-1, **kwargs):
    super(DenseToRaggedLayer, self).__init__(**kwargs)
    self.ignore_value = ignore_value

  def call(self, inputs):
    return tf.RaggedTensor.from_tensor(inputs, padding=self.ignore_value)

def embedding_dim_fn(bucket_size):
  return int(np.power(2, np.ceil(np.log(bucket_size ** 0.25)) + 3))

# Default behavior: to_wide: True, to_deep: True, transform: None
simple_sparse_config = {
  'sparse_feature1': {'bucket_size': 2100},
  'sparse_feature2': {'bucket_size': 5000000},
  'sparse_feature5': {'bucket_size': 500000},
  'sparse_feature6': {'bucket_size': 800000},
  'sparse_feature7': {'bucket_size': 800000},
  'sparse_feature8': {'bucket_size': 30000},
  'sparse_feature9': {'bucket_size': 30000},
  'sparse_feature10': {'bucket_size': 23000},
  'sparse_feature11': {'bucket_size': 23000},
  'sparse_feature12': {'bucket_size': 800000},
  'sparse_feature13': {'bucket_size': 800000},
  'sparse_feature14': {'bucket_size': 80000},
  'sparse_feature15': {'bucket_size': 80000},
  'sparse_feature16': {'bucket_size': 30000},
  'sparse_feature17': {'bucket_size': 30000},
  'sparse_feature19': {'bucket_size': 100000},
}

share_embedding_sparse_config = [
  {'name': 'ss1',
   'columns': {'sparse_feature_20': {}, 'sparse_feature_21': {}, 'sparse_feature_22': {}, 'sparse_feature_23': {}},
   'bucket_size': 220000,
   'embedding_size': 128},
  {'name': 'ss2',
   'columns': {'sparse_feature_24': {}, 'sparse_feature_25': {}, 'sparse_feature_26': {}},
   'bucket_size': 260000,
   'embedding_size': 128},
  {'name': 'ss3',
   'columns': {'sparse_feature_27': {}, 'sparse_feature_28': {}, 'sparse_feature_29': {}},
   'bucket_size': 7500000,
   'embedding_size': 64} # we can't set embedding_size to 128 in multi worker strategy because of protobuf object size limit:https://github.com/tensorflow/tensorflow/issues/45041
]

additional_dense_feature = {'label': tf.io.FixedLenFeature((1), tf.float32, 0.0),
                            'weight': tf.io.FixedLenFeature((1), tf.float32, 0.0)}

def build_model():
  parse_example_config = dict()
  wide_inputs = []
  wide_raw_inputs = []
  deep_inputs = []
  deep_raw_inputs = []
  print('[{}] Build simple sparse part...'.format(get_cur_time()))
  for feature_name, conf in simple_sparse_config.items():
    cur_input = keras.Input(shape=(None,), name=feature_name, dtype=tf.int64)
    get_parse_example_config(parse_example_config, feature_name)
    ragged_hashed_input = preprocessing.Hashing(num_bins=conf['bucket_size'], name=feature_name + '_hash')(DenseToRaggedLayer(name=feature_name + '_rag')(cur_input))
    if conf.get('to_wide', True):
      wide_raw_inputs.append(cur_input)
      wide_inputs.append(preprocessing.CategoryEncoding(max_tokens=conf['bucket_size'], output_mode='binary', sparse=True, name=feature_name + '_cat')(ragged_hashed_input))
    if conf.get('to_deep', True):
      deep_raw_inputs.append(cur_input)
      deep_inputs.append(layers.GlobalAveragePooling1D(name=feature_name + '_avg_pool')(layers.Embedding(conf['bucket_size'], embedding_dim_fn(conf['bucket_size']), name=feature_name + '_emb')(ragged_hashed_input)))
      # Bug in TF:  https://github.com/tensorflow/tensorflow/issues/45041
      # cur_emb.set_weights([np.random.random(size=(conf['bucket_size'], embedding_dim_fn(conf['bucket_size'])))])
  print('[{}] Build share embedding part...'.format(get_cur_time()))
  for conf in share_embedding_sparse_config:
    shared_embedding = layers.Embedding(conf['bucket_size'], conf['embedding_size'], name=conf['name'] + '_share_emb')
    # print('conf', conf)
    for feature_name, inner_conf in conf['columns'].items():
      cur_input = keras.Input(shape=(None,), name=feature_name, dtype=tf.int64)
      get_parse_example_config(parse_example_config, feature_name)
      ragged_hashed_input = preprocessing.Hashing(num_bins=conf['bucket_size'], name=feature_name + '_hash')(DenseToRaggedLayer(name=feature_name + '_rag')(cur_input))
      if inner_conf.get('to_wide', True):
        wide_raw_inputs.append(cur_input)
        wide_inputs.append(preprocessing.CategoryEncoding(max_tokens=conf['bucket_size'], output_mode='binary', sparse=True, name=feature_name + '_cat')(ragged_hashed_input))
      if inner_conf.get('to_deep', True):
        deep_raw_inputs.append(cur_input)
        deep_inputs.append(layers.GlobalAveragePooling1D(name=feature_name + '_avg_pool')(shared_embedding(ragged_hashed_input)))
  print('[{}] Build combine part...'.format(get_cur_time()))
  # BUILD MODEL
  wide_output = keras.experimental.LinearModel()(wide_inputs)
  deep_first = layers.Concatenate()(deep_inputs)
  dnn_model_output = keras.Sequential([keras.layers.Dense(units=512),
                                       keras.layers.Dense(units=512),
                                       keras.layers.Dense(units=512),
                                       keras.layers.Dense(units=1)])(deep_first)
  wide_model = keras.Model(inputs=wide_raw_inputs, outputs=wide_output, name='wide_model')
  deep_input_model = keras.Model(inputs=deep_raw_inputs, outputs=dnn_model_output, name='deep_model')
  print('wide model summary:')
  wide_model.summary(line_length=200)
  print('deep model summary:')
  deep_input_model.summary(line_length=200)

  wide_deep_model = tf.keras.experimental.WideDeepModel(wide_model, deep_input_model)

  parse_example_config.update(additional_dense_feature)
  return wide_deep_model, parse_example_config

biz_date = '2020-01-09-00'
train_dir = 'hdfs_train_father_directory'
eval_dir = 'hdfs_eval_father_directory'
model_dir = 'hdfs_model_dir'
BATCH_SIZE = 1024
TRAIN_EPOCHS = 1
train_files, train_sample_count = get_train_file_info(train_dir, biz_date, 7 * 24)

cluster_resolver = tf.distribute.cluster_resolver.TFConfigClusterResolver()
strategy = tf.distribute.MultiWorkerMirroredStrategy(cluster_resolver)
task_type, task_id = (strategy.cluster_resolver.task_type,
                        strategy.cluster_resolver.task_id)
print('[{}] task type: {}, task id: {}'.format(get_cur_time(), task_type, task_id))
print('[{}] Start build model...'.format(get_cur_time()))
with strategy.scope():
  global_batch_size = BATCH_SIZE * strategy.num_replicas_in_sync
  print('[{}] Global batch size is: {} ({} * {})'.format(get_cur_time(), global_batch_size, BATCH_SIZE, strategy.num_replicas_in_sync))
  model, features = build_model()
  print('[{}] Compile model...'.format(get_cur_time()))
  model.compile(optimizer=[tf.keras.optimizers.Ftrl(),
                           tf.keras.optimizers.Adagrad()])

  print('[{}] Save model structure...'.format(get_cur_time()))
  callbacks = [tf.keras.callbacks.TensorBoard(log_dir=os.path.join(model_dir, "logs"), update_freq=20, histogram_freq=1, profile_batch='2,4'),
               tf.keras.callbacks.experimental.BackupAndRestore(os.path.join(model_dir, "checkpoint"))]
  print('[{}] Start train...'.format(get_cur_time()))

  model.fit(input_fn(train_files, global_batch_size, TRAIN_EPOCHS, features),
            epochs=TRAIN_EPOCHS, callbacks=callbacks, steps_per_epoch=int(math.floor(train_sample_count * 1.0 / global_batch_size)))