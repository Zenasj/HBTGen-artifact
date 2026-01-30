from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

import multiprocessing
import os
import json
import random
import portpicker
import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing


def create_in_process_cluster(num_workers, num_ps):
  """Creates and starts local servers and returns the cluster_resolver."""
  worker_ports = [portpicker.pick_unused_port() for _ in range(num_workers)]
  ps_ports = [portpicker.pick_unused_port() for _ in range(num_ps)]

  cluster_dict = {}
  cluster_dict["worker"] = ["ip:port"]
  if num_ps > 0:
    cluster_dict["ps"] = ["ip:port"]
  print("==========================", cluster_dict["worker"], cluster_dict["ps"])
  cluster_spec = tf.train.ClusterSpec(cluster_dict)

  # Workers need some inter_ops threads to work properly.
  worker_config = tf.compat.v1.ConfigProto()
  if multiprocessing.cpu_count() < num_workers + 1:
    worker_config.inter_op_parallelism_threads = num_workers + 1
  
  os.environ["TF_CONFIG"] = json.dumps({
    "cluster": {
        "worker": cluster_dict["worker"],
        "ps": cluster_dict["ps"],
    },
    "task": {"type": "worker", "index": 0}
  })
  cluster_resolver = tf.distribute.cluster_resolver.TFConfigClusterResolver()
  return cluster_resolver

# Set the environment variable to allow reporting worker and ps failure to the
# coordinator. This is a workaround and won't be necessary in the future.
os.environ["GRPC_FAIL_FAST"] = "use_caller"

NUM_WORKERS = 1
NUM_PS = 1

cluster_resolver = create_in_process_cluster(NUM_WORKERS, NUM_PS)
os.environ["GRPC_FAIL_FAST"] = "use_caller"
if cluster_resolver.task_type in ("worker", "ps"):
    server = tf.distribute.Server(
        cluster_resolver.cluster_spec(),
        job_name=cluster_resolver.task_type,
        task_index=cluster_resolver.task_id,
        protocol=cluster_resolver.rpc_layer or "grpc",
        start=True)


strategy = tf.distribute.experimental.ParameterServerStrategy(cluster_resolver)
    
def dataset_fn(input_context):
  global_batch_size = 64
  batch_size = input_context.get_per_replica_batch_size(global_batch_size)

  x = tf.random.uniform((10, 10))
  y = tf.random.uniform((10,))

  dataset = tf.data.Dataset.from_tensor_slices((x, y)).shuffle(10).repeat()
  dataset = dataset.shard(
      input_context.num_input_pipelines,
      input_context.input_pipeline_id)
  dataset = dataset.batch(batch_size)
  dataset = dataset.prefetch(2)

  return dataset

dc = tf.keras.utils.experimental.DatasetCreator(dataset_fn)

with strategy.scope():
  model = tf.keras.models.Sequential([tf.keras.layers.Dense(10)])

model.compile(tf.keras.optimizers.SGD(), loss='mse', metrics=['accuracy'], steps_per_execution=10)


working_dir = './my_working_dir'
log_dir = os.path.join(working_dir, 'log')
ckpt_filepath = os.path.join(working_dir, 'ckpt')
backup_dir = os.path.join(working_dir, 'backup')

callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir=log_dir),
    tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_filepath),
    tf.keras.callbacks.experimental.BackupAndRestore(backup_dir=backup_dir),
    tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
]

tf.summary.trace_on()

model.fit(dc, epochs=50000, steps_per_epoch=20, callbacks=callbacks)

import time
time.sleep(100000)