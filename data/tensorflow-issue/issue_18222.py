# tf.random.uniform((B, 32, 32, 3), dtype=tf.float32) ← inferred input shape from example

import tensorflow as tf
from tensorflow.keras import layers

class MyModel(tf.keras.Model):
    """
    Custom BatchNormalization layer implementation that synchronizes statistics 
    (mean and variance) across replicas in distributed/multi-GPU settings.
    This mimics a synchronized BatchNorm behavior by using tf.distribute APIs.
    Assumptions:
    - Input shape is NHWC with channels last (e.g. (B, 32, 32, 3))
    - Sync on training, use moving averages for inference
    - Uses tf.function-compliant code for XLA compatibility
    """

    def __init__(self):
        super(MyModel, self).__init__()
        # Instantiate the custom BatchNormalization layer
        self.sync_bn = BatchNormalization(
            center=True,
            scale=True,
            epsilon=1e-3,
            name='BatchNorm',
            normaxis=-1,  # channels last
            momentum=0.99
        )

    def call(self, inputs, training=True):
        return self.sync_bn(inputs, training=training)

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Returns a random input tensor matching expected input shape NHWC: (B, H, W, C)
    # Since BatchNorm example used 32x32x3, batch size is arbitrarily 8 here
    B = 8
    H = 32
    W = 32
    C = 3
    # Provide a float32 input uniform in [0, 1)
    return tf.random.uniform((B, H, W, C), dtype=tf.float32)


class BatchNormalization(layers.Layer):
    """
    Custom Batch Normalization layer with cross-replica synchronized statistics.

    Key features:
    - Synchronize mean and variance across distributed replicas.
    - Maintain moving averages for inference.
    - Use tf.distribute.get_replica_context() for all-reduce.
    - Epsilon default set to 1e-3, momentum 0.99.
    """

    def __init__(self,
                 center=True,
                 scale=True,
                 epsilon=1e-3,
                 name='BatchNorm',
                 normaxis=-1,
                 momentum=0.99,
                 **kwargs):
        super(BatchNormalization, self).__init__(name=name, **kwargs)
        self.axis = normaxis
        self.center = center
        self.scale = scale
        self.epsilon = epsilon
        self.momentum = momentum

    def build(self, input_shape):
        # Determine feature dimension along axis
        self.feature_dim = input_shape[self.axis]
        self.axes = list(range(len(input_shape)))
        self.axes.pop(self.axis)  # all axes except feature/channel axis

        if self.scale:
            self.gamma = self.add_weight(
                shape=(self.feature_dim,),
                name='gamma',
                initializer='ones',
            )
        else:
            self.gamma = None

        if self.center:
            self.beta = self.add_weight(
                shape=(self.feature_dim,),
                name='beta',
                initializer='zeros',
            )
        else:
            self.beta = None

        # Moving mean and variance variables synchronized across replicas on read
        self.moving_mean = self.add_weight(
            name='moving_mean',
            shape=(self.feature_dim,),
            initializer=tf.initializers.zeros,
            synchronization=tf.VariableSynchronization.ON_READ,
            trainable=False,
            aggregation=tf.VariableAggregation.MEAN,
            experimental_autocast=False,
        )
        self.moving_variance = self.add_weight(
            name='moving_variance',
            shape=(self.feature_dim,),
            initializer=tf.initializers.ones,
            synchronization=tf.VariableSynchronization.ON_READ,
            trainable=False,
            aggregation=tf.VariableAggregation.MEAN,
            experimental_autocast=False,
        )
        super(BatchNormalization, self).build(input_shape)

    def _assign_moving_average(self, variable, value):
        # Update moving average for mean and variance
        return variable.assign(variable * (1.0 - self.momentum) + value * self.momentum)

    def call(self, x, training=True):
        if training:
            # Obtain replica context for distributed synchronization
            ctx = tf.distribute.get_replica_context()
            n = ctx.num_replicas_in_sync

            # Compute per-replica mean and mean of squares (variance step)
            per_replica_mean = tf.reduce_mean(x, axis=self.axes)
            per_replica_mean_sq = tf.reduce_mean(tf.square(x), axis=self.axes)

            # All-reduce sum and divide by number of replicas to get global mean and mean_sq
            mean, mean_sq = ctx.all_reduce(
                tf.distribute.ReduceOp.SUM,
                [per_replica_mean / n,
                 per_replica_mean_sq / n]
            )
            variance = mean_sq - tf.square(mean)

            # Update moving averages in a non-blocking manner
            mean_update = self._assign_moving_average(self.moving_mean, mean)
            variance_update = self._assign_moving_average(self.moving_variance, variance)
            self.add_update(mean_update)
            self.add_update(variance_update)
        else:
            mean = self.moving_mean
            variance = self.moving_variance

        # Batch normalization formula
        # scale=gamma, offset=beta, epsilon for numerical stability

        z = tf.nn.batch_normalization(
            x,
            mean=mean,
            variance=variance,
            offset=self.beta if self.beta is not None else None,
            scale=self.gamma if self.gamma is not None else None,
            variance_epsilon=self.epsilon
        )
        return z

