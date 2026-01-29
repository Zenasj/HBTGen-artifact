# tf.random.uniform((262144, 3), dtype=tf.float32)
import tensorflow as tf

class SyncBatchNormalization(tf.keras.layers.experimental.SyncBatchNormalization):
    """Custom SyncBatchNormalization to avoid NaN issues with large batch sizes on some hardware.

    This implementation overrides _calculate_mean_and_var to apply a synchronized 
    all-reduce mean and variance over replicas, with additional logging on large variance.
    
    This is based on an approach inspired by the SimCLR repo and the original issue reproduction.
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
        # Disable fused because fused SyncBatchNorm in TF 2.2 is problematic for large batches.
        super().__init__(
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
        # Compute local mean and var per replica
        shard_mean, shard_variance = super()._calculate_mean_and_var(inputs, reduction_axes, keep_dims=keep_dims)
        replica_ctx = tf.distribute.get_replica_context()
        if replica_ctx:
            # Perform all-reduce mean over all replicas for mean and variance
            group_mean, group_variance = replica_ctx.all_reduce(tf.distribute.ReduceOp.MEAN, [shard_mean, shard_variance])
            # Correct the variance by adding the mean distance variance across shards
            mean_distance = tf.math.squared_difference(tf.stop_gradient(group_mean), shard_mean)
            group_variance += replica_ctx.all_reduce(tf.distribute.ReduceOp.MEAN, mean_distance)
            # If group variance mean is large (threshold=50), print debug info
            def _print_debug():
                return tf.print(
                    f"\n{self.name} id", replica_ctx.replica_id_in_sync_group, "/",
                    replica_ctx.num_replicas_in_sync, "\n",
                    "local mean distance:", mean_distance, "mean local mean distance",
                    tf.reduce_mean(mean_distance), "\n",
                    "group var:", group_variance, "mean group var:", tf.reduce_mean(group_variance), "\n",
                    "local var:", shard_variance, "mean local var:", tf.reduce_mean(shard_variance), "\n",
                    "group mean:", group_mean, "mean group mean", tf.reduce_mean(group_mean), "\n",
                    "local mean:", shard_mean, "mean local mean", tf.reduce_mean(shard_mean), "\n",
                    "size:", tf.shape(shard_mean)
                )
            tf.cond(tf.reduce_mean(group_variance) > 50, _print_debug, lambda: tf.no_op())
            return group_mean, group_variance
        else:
            # No replica context, just use local stats
            return shard_mean, shard_variance


class MyModel(tf.keras.Model):
    """Combined model that uses 10 parallel MLP sequences, each with Dense + SyncBatchNorm + ReLU layers,
    then concatenates all outputs and passes through a final Dense layer to predict 10 classes.
    
    This replicates the provided test model from the issue.
    """
    def __init__(self):
        super().__init__()
        self.mlps = []
        for _ in range(10):
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
        outputs = []
        for mlp in self.mlps:
            x = mlp(inputs, training=training)
            outputs.append(x)
        concatenated = tf.concat(outputs, axis=-1)
        logits = self.head(concatenated)
        return logits


def my_model_function():
    # Instantiate and return the model instance
    return MyModel()


def GetInput():
    # Return a random input tensor matching the expected input shape:
    # shape = (batch_size=262144, input_features=3), dtype float32
    return tf.random.uniform((262144, 3), minval=-3, maxval=3, dtype=tf.float32)

