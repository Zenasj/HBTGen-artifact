from tensorflow import keras

synchronization=tf_variables.VariableSynchronization.ON_READ

""" MY Batchnormalization layer
ref & thanks.
https://github.com/tensorflow/tensorflow/issues/18222
https://github.com/jkyl/biggan-deep/blob/master/src/custom_layers/batch_normalization.py
https://github.com/tensorflow/community/blob/master/rfcs/20181016-replicator.md#global-batch-normalization
https://github.com/Apm5/tensorflow_2.0_tutorial/blob/master/CNN/BatchNormalization.py
"""
import tensorflow as tf
from tensorflow.keras import layers
Layer = layers.Layer

class BatchNormalization(Layer):
    def __init__(self,
                 center: bool = True,
                 scale: bool = True,
                 epsilon: float = 1e-3,
                 name: str = 'BatchNorm',
                 normaxis: int = -1,
                 momentum=0.99,
                 **kwargs):
        """
        args:
        - center: bool
        use mean statistics
        - scale: bool
        use stddev statistics
        - epsilon: float
        epsilon for zero division
        - name: str
        layer's name
        - normaxis: int
        layer's feature axis
        if data is NHWC => C (-1)
        """
        super(BatchNormalization, self).__init__(
            name=name,
            **kwargs
        )
        self.axis = normaxis
        self.center = center
        self.scale = scale
        self.epsilon = epsilon
        self.momentum = momentum

    def build(self, input_shape: list):
        """
        args:
        - input_shape: list
        example. [None, H, W, C] = [None, 32, 32, 3] (cifer 10)
        """
        self.feature_dim = input_shape[self.axis]
        self.axes = list(range(len(input_shape)))
        self.axes.pop(self.axis)
        if self.scale:
            self.gamma = self.add_weight(
                shape=(self.feature_dim,),
                name='gamma',
                initializer='ones',
            )
        if self.center:
            self.beta = self.add_weight(
                shape=(self.feature_dim,),
                name='beta',
                initializer='zeros',
            )
        self.moving_mean = self.add_weight(
            name='moving_mean',
            shape=(self.feature_dim,),
            initializer=tf.initializers.zeros,
            synchronization=tf.VariableSynchronization.ON_READ,
            trainable=False,
            aggregation=tf.VariableAggregation.MEAN,
            experimental_autocast=False)
        self.moving_variance = self.add_weight(
            name='moving_variance',
            shape=(self.feature_dim,),
            initializer=tf.initializers.ones,
            synchronization=tf.VariableSynchronization.ON_READ,
            trainable=False,
            aggregation=tf.VariableAggregation.MEAN,
            experimental_autocast=False)
        super(BatchNormalization, self).build(input_shape)

    def _assign_moving_average(self, variable: tf.Tensor, value: tf.Tensor):
        return variable.assign(variable * (1.0 - self.momentum)
                               + value * self.momentum)

    def call(self, x: tf.Tensor, training=True, **kwargs):
        if training:
            ctx = tf.distribute.get_replica_context()
            n = ctx.num_replicas_in_sync
            mean, mean_sq = ctx.all_reduce(
                tf.distribute.ReduceOp.SUM,
                [tf.reduce_mean(x, axis=self.axes) / n,
                 tf.reduce_mean(tf.square(x),
                                axis=self.axes) / n]
            )
            variance = mean_sq - mean ** 2
            mean_update = self._assign_moving_average(self.moving_mean, mean)
            variance_update = self._assign_moving_average(
                self.moving_variance, variance)
            self.add_update(mean_update)
            self.add_update(variance_update)
        else:
            mean = self.moving_mean
            variance = self.moving_variance
        z = tf.nn.batch_normalization(x, mean=mean,
                                      variance=variance,
                                      offset=self.beta,
                                      scale=self.gamma,
                                      variance_epsilon=self.epsilon)
        return z

x = tf.keras.Input([32, 32, 3])
y = BatchNormalization()(x)
model = tf.keras.Model(x, y)
model.summary()

#  Model: "model_1"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# input_11 (InputLayer)        [(None, 32, 32, 3)]       0         
# _________________________________________________________________
# BatchNorm (BatchNormalizatio (None, 32, 32, 3)         12        
# =================================================================
# Total params: 12
# Trainable params: 6
# Non-trainable params: 6
# _________________________________________________________________
model.trainable_variables
# [<tf.Variable 'BatchNorm_8/gamma:0' shape=(3,) dtype=float32, numpy=array([1., 1., 1.], dtype=float32)>, <tf.Variable 'BatchNorm_8/beta:0' shape=(3,) dtype=float32, numpy=array([0., 0., 0.], dtype=float32)>]
model.variables
# [<tf.Variable 'BatchNorm_8/gamma:0' shape=(3,) dtype=float32, numpy=array([1., 1., 1.], dtype=float32)>, <tf.Variable 'BatchNorm_8/beta:0' shape=(3,) dtype=float32, numpy=array([0., 0., 0.], dtype=float32)>, <tf.Variable 'BatchNorm_8/moving_mean:0' shape=(3,) dtype=float32, numpy=array([0., 0., 0.], dtype=float32)>, <tf.Variable 'BatchNorm_8/moving_variance:0' shape=(3,) dtype=float32, numpy=array([1., 1., 1.], dtype=float32)>]