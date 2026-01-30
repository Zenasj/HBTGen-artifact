import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf
import tensorflow.distribute as tf_dist
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.ops import array_ops
import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

def _get_current_replica_id_in_group_sync():
    replica_ctx = tf_dist.get_replica_context()
    if replica_ctx:
        replica_id = replica_ctx.replica_id_in_sync_group
    else:
        replica_id = distribute_lib.get_update_replica_id()
    if replica_id is None:
        replica_id = array_ops.constant(0, dtype=array_ops.dtypes.int32)
    return replica_id

def test(values):
    global_replica_id = _get_current_replica_id_in_group_sync()
    tf.print("global_replica_id: {}".format(global_replica_id))
    vector  = tf.zeros_like(values)
    return vector


class TestLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(TestLayer, self).__init__(**kwargs)

    # @tf.function
    def call(self, inputs, training=False):
        emb_vector = test(values = inputs)
        return emb_vector

class Demo(tf.keras.models.Model):
    def __init__(self, **kwargs):
        super(Demo, self).__init__(**kwargs)
        
        self.test_layer = TestLayer()        
        self.dense_layer = tf.keras.layers.Dense(units=1, activation=None,
                                                 kernel_initializer="ones",
                                                 bias_initializer="zeros")

    def call(self, inputs):
        vector = self.test_layer(inputs)
        logit = self.dense_layer(vector)
        return logit, vector

    def summary(self):
        inputs = tf.keras.Input(shape=(10,), dtype=tf.int64)
        model = tf.keras.models.Model(inputs=inputs, outputs=self.call(inputs))
        return model.summary()

@tf.function
def _step(inputs, labels):
    logit, vector = model(inputs)
    return logit, vector

def tf_dataset(keys, labels, batchsize, repeat):
    dataset = tf.data.Dataset.from_tensor_slices((keys, labels))
    dataset = dataset.repeat(repeat)
    dataset = dataset.batch(batchsize, drop_remainder=True)
    return dataset

def _dataset_fn(input_context):
    global_batch_size = 16384
    keys = np.ones((global_batch_size, 10))
    labels = np.random.randint(low=0, high=2, size=(global_batch_size, 1))
    replica_batch_size = input_context.get_per_replica_batch_size(global_batch_size)
    dataset = tf_dataset(keys, labels, batchsize=replica_batch_size, repeat=1)
    dataset = dataset.shard(input_context.num_input_pipelines, input_context.input_pipeline_id)
    return dataset

# Save model within MirroredStrategy scope
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = Demo()
model.compile()
model.summary()
dataset = strategy.distribute_datasets_from_function(_dataset_fn)
for i, (key_tensors, replica_labels) in enumerate(dataset):
    print("-" * 30, "step ", str(i), "-" * 30)
    logit, vector = strategy.run(_step, args=(key_tensors, replica_labels))
model.save("demo")

# Load model within MirroredStrategy scope. It works if this part is in the same script of the saving model part. But it did not when using a new script and a new scope to do this part.
with strategy.scope():
    model = tf.keras.models.load_model("demo")
dataset = strategy.distribute_datasets_from_function(_dataset_fn)
for i, (key_tensors, replica_labels) in enumerate(dataset):
    print("-" * 30, "step ", str(i), "-" * 30)
    logit, vector = strategy.run(_step, args=(key_tensors, replica_labels))



strategy = tf.distribute.MirroredStrategy(["GPU:0", "GPU:1", "GPU:2", "GPU:3"])
with strategy.scope():
    model = tf.keras.models.load_model("demo")
dataset = strategy.distribute_datasets_from_function(_dataset_fn)
for i, (key_tensors, replica_labels) in enumerate(dataset):
    print("-" * 30, "step ", str(i), "-" * 30)
    logit, vector = strategy.run(_step, args=(key_tensors, replica_labels))