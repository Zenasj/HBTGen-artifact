from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import json
import os
import sys
import time
import numpy as np
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
if "." not in sys.path:
    sys.path.insert(0, ".")

def mnist_dataset(batch_size):
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    # The `x` arrays are in uint8 and have values in the range [0, 255].
    # You need to convert them to float32 with values in the range [0, 1]
    x_train = x_train / np.float32(255)
    y_train = y_train.astype(np.int64)
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(60000).repeat().batch(batch_size)
    return train_dataset

def build_and_compile_cnn_model():
    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(28, 28)),
            tf.keras.layers.Reshape(target_shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(32, 3, activation="relu"),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(10),
        ]
    )
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
        metrics=["accuracy"],
    )
    return model

start_time = time.time()
global_batch_size = 64
multi_worker_dataset = mnist_dataset(global_batch_size)
multi_worker_model = build_and_compile_cnn_model()
multi_worker_model.fit(multi_worker_dataset, epochs=50, steps_per_epoch=70)
elapsed_time = time.time() - start_time
str_elapsed_time = time.strftime("%H : %M : %S", time.gmtime(elapsed_time))
print(">> Finished. Time elapsed: {}.".format(str_elapsed_time))

import json
import os
import sys
import time
import numpy as np
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
if "." not in sys.path:
    sys.path.insert(0, ".")

def mnist_dataset(batch_size):
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    # The `x` arrays are in uint8 and have values in the range [0, 255].
    # You need to convert them to float32 with values in the range [0, 1]
    x_train = x_train / np.float32(255)
    y_train = y_train.astype(np.int64)
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(60000).repeat().batch(batch_size)
    return train_dataset

def build_and_compile_cnn_model():
    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(28, 28)),
            tf.keras.layers.Reshape(target_shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(32, 3, activation="relu"),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(10),
        ]
    )
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
        metrics=["accuracy"],
    )
    return model

start_time = time.time()

per_worker_batch_size = 64
tf_config = json.loads(os.environ["TF_CONFIG"])
num_workers = len(tf_config["cluster"]["worker"])

strategy = tf.distribute.MultiWorkerMirroredStrategy()
global_batch_size = 64
options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
multi_worker_dataset = mnist_dataset(global_batch_size)
multi_worker_dataset_with_shrd = multi_worker_dataset.with_options(options)

with strategy.scope():
    # Model building/compiling need to be within `strategy.scope()`.
    multi_worker_model = build_and_compile_cnn_model()
multi_worker_model.fit(multi_worker_dataset_with_shrd, epochs=50, steps_per_epoch=70)
elapsed_time = time.time() - start_time
str_elapsed_time = time.strftime("%H : %M : %S", time.gmtime(elapsed_time))
print(">> Finished. Time elapsed: {}.".format(str_elapsed_time))

{"cluster": {"worker": ["xxx:2121", "yyy:2121", "zzz:2121"]}, "task": {"type": "worker", "index": 0}}
{"cluster": {"worker": ["xxx:2121", "yyy:2121", "zzz:2121"]}, "task": {"type": "worker", "index": 1}}
{"cluster": {"worker": ["xxx:2121", "yyy:2121", "zzz:2121"]}, "task": {"type": "worker", "index": 2}}

per_replica_batch_size = 64
global_batch_size = per_replica_batch_size * strategy.num_replicas_in_sync

...

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(60000).batch(global_batch_size)