import random
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
mirrored_strategy = tf.distribute.MirroredStrategy()
def get_net():
    net = tf.keras.Sequential()
    net.add(tf.keras.layers.Conv2D(filters=10,
                                   kernel_size=(3, 3)))
    net.add(tf.keras.layers.Dense(1))
    return net

data = tf.random.normal(shape=(1280, 112, 112, 3))
label = tf.random.normal(shape=(1280, ))
multi_db = tf.data.Dataset.from_tensor_slices((data, label)).batch(80)
dist_dataset = mirrored_strategy.experimental_distribute_dataset(multi_db)

with mirrored_strategy.scope():
    net = get_net()
    @tf.function
    def replica_fn(input):
        d, l = input
        return net(d)

    @tf.function
    def distribute_train_epoch(dataset):
        total_result = 0
        for x in dataset:
            per_replica_result = mirrored_strategy.experimental_run_v2(replica_fn, args=(x,))
            total_result = mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_result, axis=None)
        return total_result

    for _ in range(100):
        f = distribute_train_epoch(dist_dataset)