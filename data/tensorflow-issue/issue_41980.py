import math
import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

import numpy as np
import tensorflow as tf
from tensorflow.python.distribute import distribution_strategy_context as ds
from tensorflow.python.distribute import reduce_util
from tensorflow.python.keras.layers import normalization


class SyncBatchNormalization(normalization.BatchNormalizationBase):
    """The SyncBatchNormalization in TF 2.2 seems causing NaN issue.
    We implement this one to avoid the issue.
    See https://github.com/google-research/simclr/blob/bfe07eed7f101ab51f3360100a28690e1bfbf6ec/resnet.py#L37-L85
    """

    def __init__(self,
                 axis=-1,
                 momentum=0.99,
                 epsilon=1e-3,
                 center=True,
                 scale=True,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 moving_mean_initializer='zeros',
                 moving_variance_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 renorm=False,
                 renorm_clipping=None,
                 renorm_momentum=0.99,
                 trainable=True,
                 adjustment=None,
                 name=None,
                 **kwargs):
        # Currently we only support aggregating over the global batch size.
        super(SyncBatchNormalization, self).__init__(
            axis=axis,
            momentum=momentum,
            epsilon=epsilon,
            center=center,
            scale=scale,
            beta_initializer=beta_initializer,
            gamma_initializer=gamma_initializer,
            moving_mean_initializer=moving_mean_initializer,
            moving_variance_initializer=moving_variance_initializer,
            beta_regularizer=beta_regularizer,
            gamma_regularizer=gamma_regularizer,
            beta_constraint=beta_constraint,
            gamma_constraint=gamma_constraint,
            renorm=renorm,
            renorm_clipping=renorm_clipping,
            renorm_momentum=renorm_momentum,
            fused=False,
            trainable=trainable,
            virtual_batch_size=None,
            name=name,
            **kwargs)

    def _calculate_mean_and_var(self, inputs, reduction_axes, keep_dims):
        shard_mean, shard_variance = super(SyncBatchNormalization, self)._calculate_mean_and_var(
            inputs, reduction_axes, keep_dims=keep_dims)
        replica_ctx = ds.get_replica_context()
        if replica_ctx:
            group_mean, group_variance = replica_ctx.all_reduce(reduce_util.ReduceOp.MEAN, [shard_mean, shard_variance])
            mean_distance = tf.math.squared_difference(tf.stop_gradient(group_mean), shard_mean)
            group_variance += replica_ctx.all_reduce(reduce_util.ReduceOp.MEAN, mean_distance)
            tf.cond(tf.reduce_mean(group_variance) > 50,
                    lambda: tf.print(
                        f"\n{self.name} id", replica_ctx.replica_id_in_sync_group, "/",
                        replica_ctx.num_replicas_in_sync, "\n",
                        "local mean distance:", mean_distance, "mean local mean distance",
                        tf.reduce_mean(mean_distance), "\n",
                        "group var:", group_variance, "mean group var:", tf.reduce_mean(group_variance), "\n",
                        "local var:", shard_variance, "mean local var:", tf.reduce_mean(shard_variance), "\n",
                        "group mean:", group_mean, "mean group mean", tf.reduce_mean(group_mean), "\n",
                        "local mean:", shard_mean, "mean local mean", tf.reduce_mean(shard_mean), "\n",
                        "size:", tf.shape(shard_mean)),
                    lambda: tf.no_op()
                    )
            return group_mean, group_variance
        else:
            return shard_mean, shard_variance


class Test(tf.keras.models.Model):
    def __init__(self):
        super(Test, self).__init__()
        self.mlps = []
        for i in range(10):
            self.mlps.append(tf.keras.Sequential([
                tf.keras.layers.Dense(512),
                SyncBatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Dense(256),
                SyncBatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Dense(128),
            ]))
        self.head = tf.keras.layers.Dense(10)

    def call(self, inputs, training=None, mask=None):
        out = []
        for mlp in self.mlps:
            out.append(mlp(inputs))
        return self.head(tf.concat(out, axis=-1))


dummy_data = np.random.random((2621440, 3)).astype(np.float32) * 6 - 3
dummy_label = np.random.randint(0, 10, 2621440).astype(np.int32)
# print(dummy_label.shape)
dataset = tf.data.Dataset.from_tensor_slices((dummy_data, dummy_label)).batch(262144)

strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = Test()
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(learning_rate=0)
    )
    model.fit(dataset, epochs=10000)